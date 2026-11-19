/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Experimental pipeline handler that drives MIPI cameras through the Intel
 * camera HAL (libcamhal, "icamera" API), the same API consumed by icamerasrc.
 *
 * MVP scope: one output stream per camera, dmabuf buffers allocated by
 * libcamera and imported into the HAL, no controls (3A is run by the HAL).
 */

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <linux/videodev2.h>

#include <libcamera/base/log.h>
#include <libcamera/base/mutex.h>
#include <libcamera/base/signal.h>
#include <libcamera/base/thread.h>
#include <libcamera/base/utils.h>

#include <libcamera/camera.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer.h>
#include <libcamera/geometry.h>
#include <libcamera/property_ids.h>
#include <libcamera/stream.h>

#include "libcamera/internal/camera.h"
#include "libcamera/internal/device_enumerator.h"
#include "libcamera/internal/dma_buf_allocator.h"
#include "libcamera/internal/formats.h"
#include "libcamera/internal/framebuffer.h"
#include "libcamera/internal/media_device.h"
#include "libcamera/internal/pipeline_handler.h"
#include "libcamera/internal/request.h"
#include "libcamera/internal/v4l2_pixelformat.h"

#include <api/ICamera.h>

namespace libcamera {

LOG_DEFINE_CATEGORY(ICamHal)

namespace {

/* Consumers such as PipeWire clients ask for 8 or more buffers. */
constexpr unsigned int kMinBufferCount = 4;
constexpr unsigned int kBufferCount = 8;
constexpr unsigned int kStrideAlign = 64;
constexpr unsigned int kMaxDequeueFailures = 3;

struct SupportedMode {
	PixelFormat format;
	Size size;
	icamera::stream_t halStream;
};

} /* namespace */

class ICamHalCameraData : public Camera::Private, public Thread
{
public:
	ICamHalCameraData(PipelineHandler *pipe, int halId)
		: Camera::Private(pipe), halId_(halId)
	{
	}

	~ICamHalCameraData() override
	{
		/* Thread must not outlive the data it touches. */
		wait();
	}

	int init(const icamera::camera_info_t &info);
	const SupportedMode *findMode(const PixelFormat &format, const Size &size) const;

	void run() override;

	/* Emitted from the dequeue thread: buffer, timestamp (ns), sequence, success. */
	Signal<FrameBuffer *, uint64_t, uint32_t, bool> bufferReady;
	/* Emitted when dequeuing fails while streaming. */
	Signal<ICamHalCameraData *> deviceError;

	int halId_;
	std::string name_;
	Stream stream_;
	std::vector<SupportedMode> modes_;

	icamera::stream_t halStream_ = {};
	bool halStarted_ = false;
	std::atomic<bool> running_ = false;
	bool failed_ = false;

	Mutex mutex_;
	/* Node-based so the camera_buffer_t pointers handed to the HAL stay valid. */
	std::unordered_map<FrameBuffer *, icamera::camera_buffer_t> halBuffers_
		LIBCAMERA_TSA_GUARDED_BY(mutex_);
	std::unordered_map<icamera::camera_buffer_t *, FrameBuffer *> bufferMap_
		LIBCAMERA_TSA_GUARDED_BY(mutex_);
};

class ICamHalCameraConfiguration : public CameraConfiguration
{
public:
	ICamHalCameraConfiguration(const ICamHalCameraData *data)
		: CameraConfiguration(), data_(data)
	{
	}

	Status validate() override;

private:
	const ICamHalCameraData *data_;
};

class PipelineHandlerICamHal : public PipelineHandler
{
public:
	PipelineHandlerICamHal(CameraManager *manager)
		: PipelineHandler(manager)
	{
	}

	~PipelineHandlerICamHal();

	bool match(DeviceEnumerator *enumerator) override;

	std::unique_ptr<CameraConfiguration>
	generateConfiguration(Camera *camera, std::span<const StreamRole> roles) override;
	int configure(Camera *camera, CameraConfiguration *config) override;

	int exportFrameBuffers(Camera *camera, Stream *stream,
			       std::vector<std::unique_ptr<FrameBuffer>> *buffers) override;

	int start(Camera *camera, const ControlList *controls) override;
	void stopDevice(Camera *camera) override;
	int queueRequestDevice(Camera *camera, Request *request) override;

	bool acquireDevice(Camera *camera) override;
	void releaseDevice(Camera *camera) override;

private:
	ICamHalCameraData *cameraData(Camera *camera)
	{
		return static_cast<ICamHalCameraData *>(camera->_d());
	}

	int startHal(ICamHalCameraData *data);
	void onBufferReady(FrameBuffer *buffer, uint64_t timestamp,
			   uint32_t sequence, bool success);

	DmaBufAllocator dmaBufAllocator_{ DmaBufAllocator::DmaBufAllocatorFlag::CmaHeap |
					  DmaBufAllocator::DmaBufAllocatorFlag::SystemHeap |
					  DmaBufAllocator::DmaBufAllocatorFlag::UDmaBuf };
	void onDeviceError(ICamHalCameraData *data);

	static bool created_;
	bool resetCreated_ = false;
	bool halInitialised_ = false;
};

bool PipelineHandlerICamHal::created_ = false;

/* -----------------------------------------------------------------------------
 * Camera data
 */

int ICamHalCameraData::init(const icamera::camera_info_t &info)
{
	name_ = info.name ? info.name : "unknown";

	icamera::stream_array_t streams;
	if (!info.capability ||
	    info.capability->getSupportedStreamConfig(streams) != 0 ||
	    streams.empty()) {
		LOG(ICamHal, Error) << "HAL reports no stream configs for " << name_;
		return -EINVAL;
	}

	for (const icamera::stream_t &s : streams) {
		PixelFormat pf = V4L2PixelFormat(s.format).toPixelFormat(false);
		if (!pf.isValid() || PixelFormatInfo::info(pf).numPlanes() > 2)
			continue;

		Size size(s.width, s.height);
		if (findMode(pf, size))
			continue;

		modes_.push_back({ pf, size, s });
	}

	if (modes_.empty()) {
		LOG(ICamHal, Error) << "No usable pixel format for " << name_;
		return -EINVAL;
	}

	properties_.set(properties::Model, name_);

	/* The HAL follows the Android convention: 0 back, 1 front, 2 external. */
	int32_t location = properties::LocationExternal;
	if (info.facing == 0)
		location = properties::LocationBack;
	else if (info.facing == 1)
		location = properties::LocationFront;

	/* The profiles report facing 0 for everything; the name suffix is accurate. */
	if (name_.ends_with("-uf"))
		location = properties::LocationFront;
	else if (name_.ends_with("-wf"))
		location = properties::LocationBack;
	properties_.set(properties::Location, location);

	return 0;
}

const SupportedMode *ICamHalCameraData::findMode(const PixelFormat &format,
						 const Size &size) const
{
	for (const SupportedMode &m : modes_) {
		if (m.format == format && m.size == size)
			return &m;
	}

	return nullptr;
}

/*
 * Runs in its own thread: the HAL's dqbuf blocks until a frame is ready and
 * returns an error once the device is stopped.
 */
void ICamHalCameraData::run()
{
	unsigned int failures = 0;

	while (running_) {
		icamera::camera_buffer_t *halBuf = nullptr;
		int ret = icamera::camera_stream_dqbuf(halId_, 0, &halBuf);
		if (ret != 0 || !halBuf) {
			if (!running_)
				break;

			/* Tolerate transient errors such as timeouts. */
			LOG(ICamHal, Error) << "camera_stream_dqbuf failed: " << ret;
			if (++failures < kMaxDequeueFailures)
				continue;

			deviceError.emit(this);
			break;
		}
		failures = 0;

		FrameBuffer *buffer = nullptr;
		{
			MutexLocker locker(mutex_);
			auto it = bufferMap_.find(halBuf);
			if (it == bufferMap_.end()) {
				/* Fall back to matching on the dmabuf fd. */
				for (auto &[fb, hb] : halBuffers_) {
					if (hb.dmafd == halBuf->dmafd) {
						buffer = fb;
						break;
					}
				}
			} else {
				buffer = it->second;
			}
		}

		if (!buffer) {
			LOG(ICamHal, Error) << "Dequeued unknown HAL buffer";
			continue;
		}

		bufferReady.emit(buffer, halBuf->timestamp,
				 static_cast<uint32_t>(halBuf->sequence), true);
	}
}

/* -----------------------------------------------------------------------------
 * Configuration
 */

CameraConfiguration::Status ICamHalCameraConfiguration::validate()
{
	Status status = Valid;

	if (config_.empty())
		return Invalid;

	if (orientation != Orientation::Rotate0) {
		orientation = Orientation::Rotate0;
		status = Adjusted;
	}

	if (config_.size() > 1) {
		config_.resize(1);
		status = Adjusted;
	}

	StreamConfiguration &cfg = config_[0];
	const SupportedMode *mode = data_->findMode(cfg.pixelFormat, cfg.size);

	if (!mode) {
		/* Prefer the requested format, then the closest size. */
		const SupportedMode *best = nullptr;
		unsigned int bestDiff = UINT_MAX;

		for (int pass = 0; pass < 2 && !best; pass++) {
			for (const SupportedMode &m : data_->modes_) {
				if (pass == 0 && m.format != cfg.pixelFormat)
					continue;

				unsigned int diff =
					std::abs(static_cast<int>(m.size.width) - static_cast<int>(cfg.size.width)) +
					std::abs(static_cast<int>(m.size.height) - static_cast<int>(cfg.size.height));
				if (diff < bestDiff) {
					bestDiff = diff;
					best = &m;
				}
			}
		}

		mode = best;
		cfg.pixelFormat = mode->format;
		cfg.size = mode->size;
		status = Adjusted;
	}

	/* The HAL mode carries its aligned stride and padded buffer size. */
	cfg.stride = mode->halStream.stride;
	cfg.frameSize = mode->halStream.size;

	unsigned int count = std::clamp(cfg.bufferCount, kMinBufferCount, kBufferCount);
	if (cfg.bufferCount != count) {
		cfg.bufferCount = count;
		status = Adjusted;
	}

	if (sensorConfig) {
		sensorConfig.reset();
		status = Adjusted;
	}

	/* Colour processing is done by the HAL, which is not configurable here. */
	if (cfg.colorSpace != ColorSpace::Rec709) {
		cfg.colorSpace = ColorSpace::Rec709;
		status = Adjusted;
	}

	return status;
}

/* -----------------------------------------------------------------------------
 * Pipeline handler
 */

PipelineHandlerICamHal::~PipelineHandlerICamHal()
{
	if (resetCreated_)
		created_ = false;

	if (halInitialised_)
		icamera::camera_hal_deinit();
}

namespace {

/* HAL sensor names look like "ov02c10-uf" or "ar0234_usb_1"; entities "ov02c10 12-0036". */
bool sensorPresent(const MediaDevice *media, const std::string &name)
{
	if (!media)
		return false;

	std::string base = name.substr(0, name.find_first_of("-_"));
	for (const MediaEntity *entity : media->entities()) {
		const std::string &n = entity->name();
		if (n.size() > base.size() && n.compare(0, base.size(), base) == 0 &&
		    n[base.size()] == ' ')
			return true;
	}

	return false;
}

} /* namespace */

bool PipelineHandlerICamHal::match(DeviceEnumerator *enumerator)
{
	/* match() is called until it fails; only one instance owns the HAL. */
	if (created_)
		return false;

	/* The HAL discovers and owns the media devices itself. */
	if (icamera::camera_hal_init() != 0) {
		LOG(ICamHal, Info) << "camera_hal_init() failed, no HAL cameras";
		return false;
	}
	halInitialised_ = true;
	created_ = true;
	resetCreated_ = true;

	int num = icamera::get_number_of_cameras();
	bool registered = false;

	/*
	 * The HAL lists every sensor of its profile, present or not, and may
	 * list one sensor several times. Keep the first entry of each sensor
	 * that has an entity in the IPU6 media graph.
	 */
	DeviceMatch dm("intel-ipu6");
	std::shared_ptr<MediaDevice> media = enumerator->search(dm);
	std::set<std::string> seen;

	for (int i = 0; i < num; i++) {
		icamera::camera_info_t info = {};
		if (icamera::get_camera_info(i, info) != 0)
			continue;

		if (!info.name || !sensorPresent(media.get(), info.name) ||
		    !seen.insert(info.name).second)
			continue;

		auto data = std::make_unique<ICamHalCameraData>(this, i);
		if (data->init(info))
			continue;

		std::string id = "icamhal/" + std::to_string(i) + "-" + data->name_;
		std::set<Stream *> streams{ &data->stream_ };
		std::shared_ptr<Camera> camera =
			Camera::create(std::move(data), id, streams);
		registerCamera(std::move(camera));
		registered = true;
	}

	return registered;
}

std::unique_ptr<CameraConfiguration>
PipelineHandlerICamHal::generateConfiguration(Camera *camera,
					      std::span<const StreamRole> roles)
{
	ICamHalCameraData *data = cameraData(camera);
	auto config = std::make_unique<ICamHalCameraConfiguration>(data);

	if (roles.empty())
		return config;

	if (roles.size() > 1) {
		LOG(ICamHal, Error) << "Only one stream is supported";
		return nullptr;
	}

	if (roles[0] == StreamRole::Raw) {
		LOG(ICamHal, Error) << "Raw streams are not supported";
		return nullptr;
	}

	std::map<PixelFormat, std::vector<SizeRange>> formats;
	const SupportedMode *def = nullptr;

	for (const SupportedMode &m : data->modes_) {
		formats[m.format].emplace_back(m.size);

		/* Default: largest NV12 mode, else largest mode overall. */
		auto score = [](const SupportedMode &x) {
			return (x.format == formats::NV12 ? 1ULL << 40 : 0ULL) +
			       x.size.width * x.size.height;
		};
		if (!def || score(m) > score(*def))
			def = &m;
	}

	StreamConfiguration cfg{ StreamFormats(formats) };
	cfg.pixelFormat = def->format;
	cfg.size = def->size;
	cfg.bufferCount = kBufferCount;
	config->addConfiguration(cfg);

	config->validate();

	return config;
}

int PipelineHandlerICamHal::configure(Camera *camera, CameraConfiguration *config)
{
	ICamHalCameraData *data = cameraData(camera);
	StreamConfiguration &cfg = config->at(0);

	const SupportedMode *mode = data->findMode(cfg.pixelFormat, cfg.size);
	if (!mode)
		return -EINVAL;

	icamera::stream_t s = mode->halStream;
	s.field = V4L2_FIELD_ANY;
	s.memType = V4L2_MEMORY_DMABUF;
	s.usage = icamera::CAMERA_STREAM_PREVIEW;
	s.streamType = icamera::CAMERA_STREAM_OUTPUT;
	s.orientation = 0;

	icamera::stream_config_t list = {};
	list.num_streams = 1;
	list.streams = &s;
	list.operation_mode = icamera::CAMERA_STREAM_CONFIGURATION_MODE_AUTO;

	int ret = icamera::camera_device_config_streams(data->halId_, &list);
	if (ret != 0) {
		LOG(ICamHal, Error) << "camera_device_config_streams: " << ret;
		return -EINVAL;
	}

	/* The HAL fills in its aligned stride and buffer size. */
	data->halStream_ = s;
	cfg.stride = s.stride;
	cfg.frameSize = s.size;
	cfg.setStream(&data->stream_);

	return 0;
}

int PipelineHandlerICamHal::exportFrameBuffers([[maybe_unused]] Camera *camera, Stream *stream,
					       std::vector<std::unique_ptr<FrameBuffer>> *buffers)
{
	if (!dmaBufAllocator_.isValid())
		return -ENOBUFS;

	const StreamConfiguration &cfg = stream->configuration();
	const PixelFormatInfo &info = PixelFormatInfo::info(cfg.pixelFormat);

	std::vector<unsigned int> planeSizes;
	for (unsigned int i = 0; i < info.numPlanes(); i++)
		planeSizes.push_back(info.planeSize(cfg.size.height, i, cfg.stride));

	/* The HAL must see at least the size it asked for. */
	unsigned int total = 0;
	for (unsigned int sz : planeSizes)
		total += sz;
	if (total < cfg.frameSize)
		planeSizes.back() += cfg.frameSize - total;

	return dmaBufAllocator_.exportBuffers(cfg.bufferCount, planeSizes, buffers);
}

bool PipelineHandlerICamHal::acquireDevice(Camera *camera)
{
	ICamHalCameraData *data = cameraData(camera);

	int ret = icamera::camera_device_open(data->halId_);
	if (ret != 0) {
		LOG(ICamHal, Error) << "camera_device_open: " << ret;
		return false;
	}

	return true;
}

void PipelineHandlerICamHal::releaseDevice(Camera *camera)
{
	icamera::camera_device_close(cameraData(camera)->halId_);
}

/*
 * The HAL expects buffers to be queued before camera_device_start(), whereas
 * libcamera starts first. Defer the HAL start to the first queued request.
 */
int PipelineHandlerICamHal::start(Camera *camera,
				  [[maybe_unused]] const ControlList *controls)
{
	cameraData(camera)->failed_ = false;

	return 0;
}

int PipelineHandlerICamHal::startHal(ICamHalCameraData *data)
{
	int ret = icamera::camera_device_start(data->halId_);
	if (ret != 0) {
		LOG(ICamHal, Error) << "camera_device_start: " << ret;
		icamera::camera_device_stop(data->halId_);
		return -EIO;
	}

	data->halStarted_ = true;
	data->running_ = true;
	data->bufferReady.connect(this, &PipelineHandlerICamHal::onBufferReady);
	data->deviceError.connect(this, &PipelineHandlerICamHal::onDeviceError);
	data->Thread::start();

	return 0;
}

void PipelineHandlerICamHal::stopDevice(Camera *camera)
{
	ICamHalCameraData *data = cameraData(camera);

	data->running_ = false;

	if (data->halStarted_) {
		/* Stopping the device makes the blocked dqbuf return. */
		icamera::camera_device_stop(data->halId_);
		data->exit();
		data->wait();

		/* Deliver completions already queued to this thread. */
		thread()->dispatchMessages(Message::Type::InvokeMessage, this);
		data->bufferReady.disconnect(this);
		data->deviceError.disconnect(this);
		data->halStarted_ = false;
	}

	{
		MutexLocker locker(data->mutex_);
		data->halBuffers_.clear();
		data->bufferMap_.clear();
	}

	while (!data->queuedRequests_.empty())
		cancelRequest(data->queuedRequests_.front());
}

int PipelineHandlerICamHal::queueRequestDevice(Camera *camera, Request *request)
{
	ICamHalCameraData *data = cameraData(camera);

	FrameBuffer *buffer = request->findBuffer(&data->stream_);
	if (!buffer) {
		LOG(ICamHal, Error) << "Request has no buffer for the stream";
		return -ENOENT;
	}

	if (data->failed_)
		return -ENODEV;

	/* The HAL takes a single fd with the planes laid out contiguously. */
	std::span<const FrameBuffer::Plane> planes = buffer->planes();
	unsigned int capacity = 0;
	for (const FrameBuffer::Plane &p : planes) {
		if (p.fd.get() != planes[0].fd.get() || p.offset != capacity) {
			LOG(ICamHal, Error) << "Unsupported buffer plane layout";
			return -EINVAL;
		}
		capacity += p.length;
	}

	if (capacity < static_cast<unsigned int>(data->halStream_.size)) {
		LOG(ICamHal, Error) << "Buffer too small for the HAL stream";
		return -EINVAL;
	}

	icamera::camera_buffer_t *halBuf;
	{
		MutexLocker locker(data->mutex_);
		auto [it, inserted] = data->halBuffers_.try_emplace(buffer);
		halBuf = &it->second;
		if (inserted)
			data->bufferMap_[halBuf] = buffer;

		*halBuf = {};
		halBuf->s = data->halStream_;
		halBuf->s.memType = V4L2_MEMORY_DMABUF;
		halBuf->dmafd = planes[0].fd.get();
		/* A valid sequence and timestamp would make the HAL reprocess. */
		halBuf->sequence = -1;
		halBuf->timestamp = 0;
	}

	int ret = icamera::camera_stream_qbuf(data->halId_, &halBuf, 1);
	if (ret != 0) {
		LOG(ICamHal, Error) << "camera_stream_qbuf: " << ret;
		return -EIO;
	}

	if (!data->halStarted_)
		return startHal(data);

	return 0;
}

void PipelineHandlerICamHal::onDeviceError(ICamHalCameraData *data)
{
	LOG(ICamHal, Error) << "Capture failed on " << data->name_;

	data->failed_ = true;
	icamera::camera_device_stop(data->halId_);

	while (!data->queuedRequests_.empty())
		cancelRequest(data->queuedRequests_.front());
}

void PipelineHandlerICamHal::onBufferReady(FrameBuffer *buffer, uint64_t timestamp,
					   uint32_t sequence, bool success)
{
	Request *request = buffer->request();

	FrameMetadata &fmd = buffer->_d()->metadata();
	fmd.status = success ? FrameMetadata::FrameSuccess : FrameMetadata::FrameError;
	fmd.sequence = sequence;
	fmd.timestamp = timestamp;

	/* The HAL writes one contiguous image; report it all on the plane(s). */
	std::span<const FrameBuffer::Plane> planes = buffer->planes();
	for (const auto [i, p] : utils::enumerate(planes))
		fmd.planes()[i].bytesused = p.length;

	request->_d()->metadata().set(controls::SensorTimestamp,
				      static_cast<int64_t>(timestamp));

	if (completeBuffer(request, buffer))
		completeRequest(request);
}

REGISTER_PIPELINE_HANDLER(PipelineHandlerICamHal, "icamhal")

} /* namespace libcamera */
