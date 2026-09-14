@file:androidx.annotation.OptIn(androidx.camera.view.TransformExperimental::class)

package org.aurora2w.app

import android.graphics.*
import androidx.camera.core.ImageProxy
import androidx.camera.view.transform.*

/** Coordinates are pixel edges in the cropped, upright analysis bitmap.
 * Both use cases must share a CameraX ViewPort; neither path mirrors the rear camera.
 */
object CameraFrameTransform {
    fun source(proxy: ImageProxy): OutputTransform = ImageProxyTransformFactory().apply {
        isUsingCropRect=true;isUsingRotationDegrees=true
    }.getOutputTransform(proxy)
    fun upright(full: Bitmap,crop: Rect,rotation: Int): Bitmap =
        Bitmap.createBitmap(full,crop.left,crop.top,crop.width(),crop.height(),
            Matrix().apply {postRotate(rotation.toFloat())},true)
    fun toPreview(source: OutputTransform,target: OutputTransform): Matrix =
        Matrix().also {CoordinateTransform(source,target).transform(it)}
}
