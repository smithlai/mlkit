/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.kotlin.dashcam

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import com.google.android.odml.image.MlImage
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.VisionProcessorBase
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.MoveNetMultiPose
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.TrackerType
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.Type
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.Device
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.Person
import com.google.mlkit.vision.demo.kotlin.dashcam.overlay.FaceMeshGraphic
import com.google.mlkit.vision.demo.kotlin.dashcam.overlay.MoveNetGraphic
import com.google.mlkit.vision.demo.kotlin.dashcam.overlay.ObjectGraphic
import com.google.mlkit.vision.demo.kotlin.dashcam.overlay.PoseGraphic
import com.google.mlkit.vision.demo.kotlin.dashcam.poseclassification.PoseClassifierProcessor
import com.google.mlkit.vision.facemesh.FaceMesh
import com.google.mlkit.vision.facemesh.FaceMeshDetection
import com.google.mlkit.vision.facemesh.FaceMeshDetector
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import java.io.IOException
import java.util.concurrent.Executor
import java.util.concurrent.Executors


/** A processor to run pose detector. */
class DashcamMLProcessor(
  private val context: Context,
  poseOptions: PoseDetectorOptionsBase,
  private val showInFrameLikelihood: Boolean,
  private val visualizeZ: Boolean,
  private val rescaleZForVisualization: Boolean,
  private val bPoseClassification: Boolean,
  private val isStreamMode: Boolean,
  odOptions: ObjectDetectorOptions,
  fmOptions: FaceMeshDetectorOptions
) : VisionProcessorBase<DashcamMLProcessor.CompoundDtection>(context) {

  private val poseDetector: PoseDetector
  private val odDetector: ObjectDetector
  private val fmDetector: FaceMeshDetector
  private val movenetDetector: MoveNetMultiPose
  private val poseClassificationExecutor: Executor

  private var poseClassifierProcessor: PoseClassifierProcessor? = null



  /** Internal class to hold Pose and classification results. */
  class CompoundDtection(val pose: Pose?, val classificationResult: List<String>?, val detectedObjects: List<DetectedObject>?, val detectedFaceMeshs: List<FaceMesh>?, val personlist:List<Person>?)

  init {
    // (MediaPipe Blazepose) https://developers.google.com/ml-kit/vision/pose-detection#under_the_hood
    poseDetector = PoseDetection.getClient(poseOptions)
    odDetector = ObjectDetection.getClient(odOptions)
    poseClassificationExecutor = Executors.newSingleThreadExecutor()

    // https://developers.google.com/ml-kit/vision/face-mesh-detection/concepts
    fmDetector = FaceMeshDetection.getClient(fmOptions)

    // This movement copied from:
    // https://github.com/smithlai/tensorflow-examples/tree/master/lite/examples/pose_estimation
    movenetDetector = MoveNetMultiPose.create(
      context,
      Device.NNAPI,
      Type.Dynamic
    ).apply {
      setTracker(TrackerType.BOUNDING_BOX)
    }
  }

  override fun stop() {
    super.stop()
    try {
      poseDetector.close()
      odDetector.close()
      fmDetector.close()
      movenetDetector.close()
    } catch (e: IOException) {
      Log.e(
        TAG,
        "Exception thrown while trying to close detector!",
        e
      )
    }

  }

  override fun detectInImage(image: InputImage, bmp: Bitmap?): Task<CompoundDtection> {
    val task1 = poseDetector.process(image)
    val task2 = odDetector.process(image)
    val task3 = fmDetector.process(image)
    var task4: Task<List<Person>>? = null
    bmp?.let{
//      task4 = movenetDetector.process(bmp)
    }
    return waitTasks(task1, task2, task3, task4)
  }

  override fun detectInImage(image: MlImage, bmp: Bitmap?): Task<CompoundDtection> {
//    // Image converter examples:
//    val mediaimage = MediaImageExtractor.extract(image)
//    val bitmap:Bitmap = BitmapExtractor.extract(image)
//    val tensorImage = MlImageAdapter.createTensorImageFrom(image)
//    val tensorImage = TensorImage.fromBitmap(image.bitmapInternal)
    var task1: Task<Pose>? = null
    var task2: Task<List<DetectedObject>>? = null
    var task3: Task<List<FaceMesh>>? = null
    var task4: Task<List<Person>>? = null

    task1 = poseDetector.process(image)
    task2 = odDetector.process(image)
    task3 = fmDetector.process(image)

    bmp?.let{
      task4 = movenetDetector.process(it)
    }

    return waitTasks(task1, task2, task3, task4)
  }
  private fun waitTasks(taskPose: Task<Pose>?, taskOD: Task<List<DetectedObject>>?, taskFM:Task<List<FaceMesh>>?, task4: Task<List<Person>>?) : Task<CompoundDtection>{
    //to return Task Type for detectInImage and processImageProxy
    val taskCompletionSource = TaskCompletionSource<CompoundDtection>()

    var classificationResult: List<String> = ArrayList()
    taskPose?.continueWith(
      poseClassificationExecutor,
      { task ->
        val pose = task.getResult()

        if (bPoseClassification) {
          if (poseClassifierProcessor == null) {
            poseClassifierProcessor = PoseClassifierProcessor(context, isStreamMode)
          }
          classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
        }
        pose
      }
    )

    var taskmap = mutableMapOf<String, Task<Any>>()
    taskPose?.let { taskmap.put("pose", it as Task<Any>) }
    taskOD?.let { taskmap.put("OD", it as Task<Any>) }
    taskFM?.let { taskmap.put("FM", it as Task<Any>) }
    task4?.let { taskmap.put("task4", it as Task<Any>) }

    var pose:Pose? = null
    var detectedObjects:List<DetectedObject>? = null
    var detectedFacemeshs:List<FaceMesh>? = null
    var personlist:List<Person>? = null
    while (taskmap.size > 0){
//      Log.e("xxxxx", "1taskmap.size:" + taskmap.size)
      val iterator: MutableIterator<Map.Entry<String, Task<Any>>> = taskmap.entries.iterator()
      while (iterator.hasNext()) {
        val entry = iterator.next()
        val k = entry.key
        val t = entry.value
        if (t.isComplete) {
          when (k) {
            "pose" -> pose = taskPose?.getResult() // <Pose>
            "OD" -> detectedObjects = taskOD?.getResult() // <List<DetectedObject>>
            "FM" -> detectedFacemeshs = taskFM?.getResult() // List<FaceMesh>
            "task4" -> task4?.let {
              personlist = it.getResult() // List<Person>
            }
          }
          iterator.remove()
//          Log.e("xxxxx", k + "... complete, taskmap.size:" + taskmap.size)
        }
      }
    }
//    Log.e("xxxxx", "2taskmap.size:" + 0)





    taskCompletionSource.setResult(CompoundDtection(pose, classificationResult, detectedObjects, detectedFacemeshs, personlist))
    return taskCompletionSource.getTask()

  }

  override fun onSuccess(
    results: CompoundDtection,
    graphicOverlay: GraphicOverlay
  ) {
    if ((null != results.pose) && (null != results.classificationResult)) {
      results.pose?.let {
        graphicOverlay.add(
          PoseGraphic(
            graphicOverlay,
            results.pose,
            showInFrameLikelihood,
            visualizeZ,
            rescaleZForVisualization,
            results.classificationResult
          )
        )
      }
    }
    results.detectedObjects?.let {
      Log.e("results.detectedObjects", it.toString())
      for (detectedObject in it) {
        graphicOverlay.add(ObjectGraphic(graphicOverlay, detectedObject))
      }
    }
    results.detectedFaceMeshs?.let {
      for (facemesh in it) {
        graphicOverlay.add(FaceMeshGraphic(graphicOverlay, facemesh))
      }
    }
    results.personlist?.let{
      graphicOverlay.add(MoveNetGraphic(graphicOverlay, it))
    }
  }

  override fun onFailure(e: Exception) {
    Log.e(TAG, "Pose detection failed!", e)
  }

  override fun isMlImageEnabled(context: Context?): Boolean {
    // Use MlImage in Pose Detection by default, change it to OFF to switch to InputImage.
    return true
  }

  companion object {
    private val TAG = "DashcamMLProcessor"
  }

}
