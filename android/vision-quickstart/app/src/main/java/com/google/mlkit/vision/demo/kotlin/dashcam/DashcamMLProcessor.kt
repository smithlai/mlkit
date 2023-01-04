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
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor
import com.google.mlkit.vision.demo.kotlin.VisionProcessorBase
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
  odOptions: ObjectDetectorOptions
) : VisionProcessorBase<DashcamMLProcessor.CompoundDtection>(context) {

  private val poseDetector: PoseDetector
  private val odDetector: ObjectDetector
  private val poseClassificationExecutor: Executor

  private var poseClassifierProcessor: PoseClassifierProcessor? = null

  /** Internal class to hold Pose and classification results. */
  class CompoundDtection(val pose: Pose, val classificationResult: List<String>)

  init {
    poseDetector = PoseDetection.getClient(poseOptions)
    odDetector = ObjectDetection.getClient(odOptions)
    poseClassificationExecutor = Executors.newSingleThreadExecutor()
  }

  override fun stop() {
    super.stop()
    try {
      poseDetector.close()
      odDetector.close()
    } catch (e: IOException) {
      Log.e(
        TAG,
        "Exception thrown while trying to close detector!",
        e
      )
    }

  }

  override fun detectInImage(image: InputImage): Task<CompoundDtection> {
    Log.e("aaaa", "11111111111111")
//    return odDetector.process(image)
    return poseDetector
      .process(image)
      .continueWith(
        poseClassificationExecutor,
        { task ->
          val pose = task.getResult()
          var classificationResult: List<String> = ArrayList()
          if (bPoseClassification) {
            if (poseClassifierProcessor == null) {
              poseClassifierProcessor = PoseClassifierProcessor(context, isStreamMode)
            }
            classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
          }
          CompoundDtection(pose, classificationResult)
        }
      )
  }

  override fun detectInImage(image: MlImage): Task<CompoundDtection> {
    //to return Task Type
    val taskCompletionSource = TaskCompletionSource<CompoundDtection>()
    //pose
    lateinit var pose: Pose
    var classificationResult: List<String> = ArrayList()

    val task1 = poseDetector.process(image).continueWith(
      poseClassificationExecutor,
      { task ->
        pose = task.getResult()

        if (bPoseClassification) {
          if (poseClassifierProcessor == null) {
            poseClassifierProcessor = PoseClassifierProcessor(context, isStreamMode)
          }
          classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
        }
        CompoundDtection(pose, classificationResult)
      }
    )
    Log.e("aaaa", "333333")
    for (t in listOf(task1)){
      while (!t.isComplete){
        try {
            Log.e("xxxxx", t.isComplete.toString())
            Thread.sleep(5)
        }catch (e: Exception){
//          taskCompletionSource.setException(e)
        }
      }
    }
//    Log.e("aaaa", "4444444444444")
//    Log.e("aaaa", task1.getResult().toString())
    taskCompletionSource.setResult(CompoundDtection(pose, classificationResult))
    return taskCompletionSource.getTask()

  }

  override fun onSuccess(
    poseWithClassification: CompoundDtection,
    graphicOverlay: GraphicOverlay
  ) {
    graphicOverlay.add(
      DashcamMLGraphic(
        graphicOverlay,
        poseWithClassification.pose,
        showInFrameLikelihood,
        visualizeZ,
        rescaleZForVisualization,
        poseWithClassification.classificationResult
      )
    )
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
