/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

/*
* https://ithelp.ithome.com.tw/articles/10276714
*
* MoveNet：最先進的姿勢估計模型有兩種版本：Lighting 和 Thunder。
* PoseNet：2017 年發布的上一代姿態估計模型。
* MoveNet 有兩種版本：
* MoveNet.Lightning 比 Thunder 版本更小、更快但準確度較低。它可以在現代智能手機上實時運行。
* MoveNet.Thunder 是更準確的版本，但也比 Lightning 更大更慢。它對於需要更高準確性的用例很有用。
* MoveNet 在各種數據集上的表現都優於 PoseNet，尤其是在帶有健身動作圖像的圖像中。因此，我們建議在 PoseNet 上使用 MoveNet。
*
* */
package com.google.mlkit.vision.demo.kotlin.dashcam.customtflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.BodyPart
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.Device
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.KeyPoint
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.Person
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.tracker.AbstractTracker
import org.tensorflow.lite.examples.poseestimation.tracker.BoundingBoxTracker
import org.tensorflow.lite.examples.poseestimation.tracker.KeyPointsTracker
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.ceil

class MoveNetMultiPose(
    private val interpreter: Interpreter,
    private val type: Type,
    private val gpuDelegate: GpuDelegate?,
) : PoseDetector {
    private val outputShape = interpreter.getOutputTensor(0).shape()
    private val inputShape = interpreter.getInputTensor(0).shape()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    private var targetWidth: Int = 0
    private var targetHeight: Int = 0
    private var scaleHeight: Int = 0
    private var scaleWidth: Int = 0
    private var lastInferenceTimeNanos: Long = -1
    private var tracker: AbstractTracker? = null

    companion object {
        public  const val TAG = "MovenetMultiPose"
        private const val DYNAMIC_MODEL_TARGET_INPUT_SIZE = 256
        private const val SHAPE_MULTIPLE = 32.0
        private const val DETECTION_THRESHOLD = 0.11
        private const val DETECTION_SCORE_INDEX = 55
        private const val BOUNDING_BOX_Y_MIN_INDEX = 51
        private const val BOUNDING_BOX_X_MIN_INDEX = 52
        private const val BOUNDING_BOX_Y_MAX_INDEX = 53
        private const val BOUNDING_BOX_X_MAX_INDEX = 54
        private const val KEYPOINT_COUNT = 17
        private const val OUTPUTS_COUNT_PER_KEYPOINT = 3
        private const val CPU_NUM_THREADS = 4

        // allow specifying model type.
        fun create(
            context: Context,
            device: Device,
            type: Type,
        ): MoveNetMultiPose {
            val options = Interpreter.Options()
            var gpuDelegate: GpuDelegate? = null
            when (device) {
                Device.CPU -> {
                    options.setNumThreads(CPU_NUM_THREADS)
                }
                Device.GPU -> {
                    // only fixed model support Gpu delegate option.
                    if (type == Type.Fixed) {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                    }
                }
                else -> {
                    // nothing to do
                }
            }
            return MoveNetMultiPose(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        if (type == Type.Dynamic)
                            "custom_models/movenet_multipose_fp16.tflite" else ""
                        //@TODO: (khanhlvg) Add support for fixed shape model if it's released.
                    ), options
                ), type, gpuDelegate
            )
        }
    }

    fun process(image: Bitmap): Task<List<Person>> {
        val t = TaskCompletionSource<List<Person>>();
        val job = GlobalScope.launch(Dispatchers.IO) {
            val startMs = SystemClock.elapsedRealtime()
              // TODO: if I sleep too long here, the object detection model will be affected (cannot detect anything).
            val personList = estimatePoses(image)
//            Thread.sleep(300L)
//            val personList = listOf<Person>()
            val endMs = SystemClock.elapsedRealtime()
            Log.e(TAG, "MoveNet Inference:" + (endMs-startMs)+ " ms")

            t.setResult(personList)
        }
        return t.task
    }

    /**
     * Convert x and y coordinates ([0-1]) returns from the TFlite model
     * to the coordinates corresponding to the input image.
     */
    private fun resizeKeypoint(x: Float, y: Float): PointF {
        return PointF(resizeX(x), resizeY(y))
    }

    private fun resizeX(x: Float): Float {
        return if (imageWidth > imageHeight) {
            val ratioWidth = imageWidth.toFloat() / targetWidth
            x * targetWidth * ratioWidth
        } else {
            val detectedWidth =
                if (type == Type.Dynamic) targetWidth else inputShape[2]
            val paddingWidth = detectedWidth - scaleWidth
            val ratioWidth = imageWidth.toFloat() / scaleWidth
            (x * detectedWidth - paddingWidth / 2f) * ratioWidth
        }
    }

    private fun resizeY(y: Float): Float {
        return if (imageWidth > imageHeight) {
            val detectedHeight =
                if (type == Type.Dynamic) targetHeight else inputShape[1]
            val paddingHeight = detectedHeight - scaleHeight
            val ratioHeight = imageHeight.toFloat() / scaleHeight
            (y * detectedHeight - paddingHeight / 2f) * ratioHeight
        } else {
            val ratioHeight = imageHeight.toFloat() / targetHeight
            y * targetHeight * ratioHeight
        }
    }

    /**
     * Prepare input image for detection
     * https://storage.googleapis.com/movenet/MoveNet.MultiPose%20Model%20Card.pdf
     * """
     * H and W need to be a multiple of 32 and can be determined at run time.
     * A recommended way to prepare the input image tensor is to resize the image such
     * that its larger side is equal to 256 pixels while keeping the image’s original aspect ratio.
     * """
     */
    private fun processInputTensor(bitmap: Bitmap): TensorImage {
        imageWidth = bitmap.width
        imageHeight = bitmap.height

        // if model type is fixed. get input size from input shape.
        // else, set to 256
        val inputSizeHeight =
            if (type == Type.Dynamic) DYNAMIC_MODEL_TARGET_INPUT_SIZE else inputShape[1]
        val inputSizeWidth =
            if (type == Type.Dynamic) DYNAMIC_MODEL_TARGET_INPUT_SIZE else inputShape[2]


        // 1. resizeOp: scale with max size 256 and keep the original ratio (but )
        val resizeOp: ImageOperator
        if (imageWidth > imageHeight) {
            val scale = inputSizeWidth / imageWidth.toFloat()
            targetWidth = inputSizeWidth
            scaleHeight = ceil(imageHeight * scale).toInt()

            resizeOp = ResizeOp(scaleHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR)
            targetHeight = (ceil((scaleHeight / SHAPE_MULTIPLE)) * SHAPE_MULTIPLE).toInt()
            //resizeOp = ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR)
        } else {
            val scale = inputSizeHeight / imageHeight.toFloat()
            targetHeight = inputSizeHeight
            scaleWidth = ceil(imageWidth * scale).toInt()

            resizeOp = ResizeOp(targetHeight, scaleWidth, ResizeOp.ResizeMethod.BILINEAR)
            targetWidth = (ceil((scaleWidth / SHAPE_MULTIPLE)) * SHAPE_MULTIPLE).toInt()
            //resizeOp = ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR)
        }

        // 2. resizeWithCropOrPad: Resize bitmap to max size 256*256
        val resizeWithCropOrPad = if (type == Type.Dynamic) ResizeWithCropOrPadOp(
            targetHeight,
            targetWidth
        ) else ResizeWithCropOrPadOp(
            inputSizeHeight,
            inputSizeWidth
        )
        val imageProcessor = ImageProcessor.Builder().apply {
            add(resizeOp)
            add(resizeWithCropOrPad)
        }.build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    /**
     * Run tracker (if available) and process the output.
     */
    private fun postProcess(modelOutput: FloatArray): List<Person> {
        val persons = mutableListOf<Person>()
        for (idx in modelOutput.indices step outputShape[2]) {
            val personScore = modelOutput[idx + DETECTION_SCORE_INDEX]
            if (personScore < DETECTION_THRESHOLD) continue
            val positions = modelOutput.copyOfRange(idx, idx + 51)
            val keyPoints = mutableListOf<KeyPoint>()
            for (i in 0 until KEYPOINT_COUNT) {
                val y = positions[i * OUTPUTS_COUNT_PER_KEYPOINT]
                val x = positions[i * OUTPUTS_COUNT_PER_KEYPOINT + 1]
                val score = positions[i * OUTPUTS_COUNT_PER_KEYPOINT + 2]
                keyPoints.add(KeyPoint(BodyPart.fromInt(i), PointF(x, y), score))
            }
            val yMin = modelOutput[idx + BOUNDING_BOX_Y_MIN_INDEX]
            val xMin = modelOutput[idx + BOUNDING_BOX_X_MIN_INDEX]
            val yMax = modelOutput[idx + BOUNDING_BOX_Y_MAX_INDEX]
            val xMax = modelOutput[idx + BOUNDING_BOX_X_MAX_INDEX]
            val boundingBox = RectF(xMin, yMin, xMax, yMax)
            persons.add(
                Person(
                    keyPoints = keyPoints,
                    boundingBox = boundingBox,
                    score = personScore
                )
            )
        }

        if (persons.isEmpty()) return emptyList()

        if (tracker == null) {
            // no tracker, just resize keypoints
            persons.forEach {
                it.keyPoints.forEach { key ->
                    key.coordinate = resizeKeypoint(key.coordinate.x, key.coordinate.y)
                }
            }
            return persons
        } else {
            val trackPersons = mutableListOf<Person>()
            tracker?.apply(persons, System.currentTimeMillis() * 1000)?.forEach {
                val resizeKeyPoint = mutableListOf<KeyPoint>()
                it.keyPoints.forEach { key ->
                    resizeKeyPoint.add(
                        KeyPoint(
                            key.bodyPart,
                            resizeKeypoint(key.coordinate.x, key.coordinate.y),
                            key.score
                        )
                    )
                }

                var resizeBoundingBox: RectF? = null
                it.boundingBox?.let { boundingBox ->
                    resizeBoundingBox = RectF(
                        resizeX(boundingBox.left),
                        resizeY(boundingBox.top),
                        resizeX(boundingBox.right),
                        resizeY(boundingBox.bottom)
                    )
                }
                trackPersons.add(Person(it.id, resizeKeyPoint, resizeBoundingBox, it.score))
            }
            return trackPersons
        }
    }

    /**
     * Create and set tracker.
     */
    fun setTracker(trackerType: TrackerType) {
        tracker = when (trackerType) {
            TrackerType.BOUNDING_BOX -> {
                BoundingBoxTracker()
            }
            TrackerType.KEYPOINTS -> {
                KeyPointsTracker()
            }
            TrackerType.OFF -> {
                null
            }
        }
    }

    /**
     * Run TFlite model and Returns a list of "Person" corresponding to the input image.
     */
    override fun estimatePoses(bitmap: Bitmap): List<Person> {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
//        val tmp_bmp = bitmap.copy(bitmap.getConfig(), true);
        val inputTensor = processInputTensor(bitmap)
        val outputTensor = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        // if model is dynamic, resize input before run interpreter
        if (type == Type.Dynamic) {
            val inputShape = intArrayOf(1).plus(inputTensor.tensorBuffer.shape)
            interpreter.resizeInput(0, inputShape, true)
            interpreter.allocateTensors()
        }
        interpreter.run(inputTensor.buffer, outputTensor.buffer.rewind())

        val processedPerson = postProcess(outputTensor.floatArray)
        lastInferenceTimeNanos =
            SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        return processedPerson
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos

    /**
     * Close all resources when not in use.
     */
    override fun close() {
        gpuDelegate?.close()
        interpreter.close()
        tracker = null
    }
}

enum class Type {
    Dynamic, Fixed
}

enum class TrackerType {
    OFF, BOUNDING_BOX, KEYPOINTS
}
