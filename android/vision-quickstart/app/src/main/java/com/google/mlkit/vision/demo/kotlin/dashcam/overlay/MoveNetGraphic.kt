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

package com.google.mlkit.vision.demo.kotlin.dashcam.overlay

import android.graphics.*
import android.os.Build
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.Person
import android.graphics.PointF
import com.google.mlkit.vision.demo.kotlin.dashcam.customtflite.movenetdata.data.BodyPart
import kotlin.math.max
import kotlin.math.min

/** Draw the detected pose in preview. */
class MoveNetGraphic
internal constructor(
  overlay: GraphicOverlay,
  private val persons: List<Person>
) : Graphic(overlay) {

  init {
  }

  // Draw line and point indicate body pose
  fun drawBodyKeypoints(
    canvas: Canvas,
    persons: List<Person>,
  ){
    val isTrackerEnabled: Boolean = true
    val paintCircle = Paint().apply {
      strokeWidth = CIRCLE_RADIUS
      color = Color.RED
      style = Paint.Style.FILL
    }
    val paintLine = Paint().apply {
      strokeWidth = LINE_WIDTH
      color = Color.RED
      style = Paint.Style.STROKE
    }

    val paintText = Paint().apply {
      textSize = PERSON_ID_TEXT_SIZE
      color = Color.BLUE
      textAlign = Paint.Align.LEFT
    }

    persons.forEach { person ->
      // draw person id if tracker is enable
      if (isTrackerEnabled) {
        // 1. Boundingbox (calc from one of 2 trackers)
        person.boundingBox?.let {
          val rect = RectF(it)
          val x0 = translateX(rect.left)
          val x1 = translateX(rect.right)
          rect.left = min(x0, x1)
          rect.right = max(x0, x1)
          rect.top = translateY(rect.top)
          rect.bottom = translateY(rect.bottom)

          val personIdX = max(0f, rect.left)
          val personIdY = max(0f, rect.top)

          canvas.drawText(
            "Movenet:"+person.id.toString(),
            personIdX,
            personIdY - PERSON_ID_MARGIN,
            paintText
          )
          canvas.drawRect(rect, paintLine)
        }
      }
      // 2. Bodyline
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        bodyJoints.forEach {
          val pointA = PointF(person.keyPoints[it.first.position].coordinate)
          val pointB = PointF(person.keyPoints[it.second.position].coordinate)
          pointA.apply {
            x = translateX(x)
            y = translateY(y)
          }

          pointB.apply {
            x = translateX(x)
            y = translateY(y)
          }
          canvas.drawLine(pointA.x, pointA.y, pointB.x, pointB.y, paintLine)
        }
      }
      //3. bodypoint
      person.keyPoints.forEach { point ->
        canvas.drawCircle(
          translateX(point.coordinate.x),
          translateY(point.coordinate.y),
          CIRCLE_RADIUS,
          paintCircle
        )
      }
    }
  }

  override fun draw(canvas: Canvas) {
    drawBodyKeypoints(
      canvas,
      persons.filter { it.score > MIN_CONFIDENCE }
    )
  }


  companion object {
    private const val MIN_CONFIDENCE = .2f

    /** Radius of circle used to draw keypoints.  */
    private const val CIRCLE_RADIUS = 6f

    /** Width of line used to connected two keypoints.  */
    private const val LINE_WIDTH = 4f

    /** The text size of the person id that will be displayed when the tracker is available.  */
    private const val PERSON_ID_TEXT_SIZE = 30f

    /** Distance from person id to the nose keypoint.  */
    private const val PERSON_ID_MARGIN = 6f

    /** Pair of keypoints to draw lines between.  */
    private val bodyJoints = listOf(
      Pair(BodyPart.NOSE, BodyPart.LEFT_EYE),
      Pair(BodyPart.NOSE, BodyPart.RIGHT_EYE),
      Pair(BodyPart.LEFT_EYE, BodyPart.LEFT_EAR),
      Pair(BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR),
      Pair(BodyPart.NOSE, BodyPart.LEFT_SHOULDER),
      Pair(BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
      Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
      Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
      Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
      Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
      Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
      Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
      Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
      Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
      Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
      Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
      Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
      Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
    )
  }

}
