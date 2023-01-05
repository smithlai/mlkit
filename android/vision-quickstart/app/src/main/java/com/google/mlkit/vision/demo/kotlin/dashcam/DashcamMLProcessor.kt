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
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import com.google.android.odml.image.MlImage
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.VisionProcessorBase
import com.google.mlkit.vision.demo.kotlin.dashcam.poseclassification.PoseClassifierProcessor
import com.google.mlkit.vision.face.Face
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
import java.lang.Thread
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
  private val poseClassificationExecutor: Executor

  private var poseClassifierProcessor: PoseClassifierProcessor? = null



  /** Internal class to hold Pose and classification results. */
  class CompoundDtection(val pose: Pose, val classificationResult: List<String>, val detectedObjects: List<DetectedObject>, val detectedFaceMeshs: List<FaceMesh>)

  init {
    poseDetector = PoseDetection.getClient(poseOptions)
    odDetector = ObjectDetection.getClient(odOptions)
    poseClassificationExecutor = Executors.newSingleThreadExecutor()

    fmDetector = FaceMeshDetection.getClient(fmOptions)

  }

  override fun stop() {
    super.stop()
    try {
      poseDetector.close()
      odDetector.close()
      fmDetector.close()
    } catch (e: IOException) {
      Log.e(
        TAG,
        "Exception thrown while trying to close detector!",
        e
      )
    }

  }

  override fun detectInImage(image: InputImage): Task<CompoundDtection> {
    val task1 = poseDetector.process(image)
    val task2 = odDetector.process(image)
    val task3 = fmDetector.process(image)
    return waitTasks(task1, task2, task3)
  }

  override fun detectInImage(image: MlImage): Task<CompoundDtection> {

    val task1 = poseDetector.process(image)
    val task2 = odDetector.process(image)
    val task3 = fmDetector.process(image)
    return waitTasks(task1, task2, task3)
  }
  private fun waitTasks(taskPose: Task<Pose>, taskOD: Task<List<DetectedObject>>, taskFM:Task<List<FaceMesh>>) : Task<CompoundDtection>{
    //to return Task Type for detectInImage and processImageProxy
    val taskCompletionSource = TaskCompletionSource<CompoundDtection>()

    var classificationResult: List<String> = ArrayList()
    taskPose.continueWith(
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


    for ((k,t) in mapOf("pose" to taskPose,"OD" to taskOD, "FM" to taskFM)){
      while (!t.isComplete){
        try {
//          Log.e("xxxxx", k + "... waiting complete")
          Thread.sleep(5)
        }catch (e: Exception){
//          taskCompletionSource.setException(e)
        }
      }
    }
    val pose = taskPose.getResult() // <Pose>
    val detectedObjects = taskOD.getResult() // <List<DetectedObject>>
    val detectedFacemeshs = taskFM.getResult() // List<FaceMesh>
    taskCompletionSource.setResult(CompoundDtection(pose, classificationResult, detectedObjects, detectedFacemeshs))
    return taskCompletionSource.getTask()

  }

  override fun onSuccess(
    results: CompoundDtection,
    graphicOverlay: GraphicOverlay
  ) {
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
    for (detectedObject in results.detectedObjects) {
      graphicOverlay.add(ObjectGraphic(graphicOverlay, detectedObject))
    }

    for (facemesh in results.detectedFaceMeshs) {
      graphicOverlay.add(FaceMeshGraphic(graphicOverlay, facemesh))
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
