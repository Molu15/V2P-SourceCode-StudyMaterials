package com.example.smombie_bridge

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.LinearLayout

object OverlayHelper {
    private var overlayView: FrameLayout? = null
    private var edgeOverlayView: FrameLayout? = null
    private var windowManager: WindowManager? = null

    // direction: "left" | "right" | null (null = pre-study, no arrow)
    fun showOverlay(context: Context, direction: String? = null) {
        Handler(Looper.getMainLooper()).post {
            try {
                val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                windowManager = wm

                // Falls bereits aktiv, entfernen
                overlayView?.let {
                    try { wm.removeView(it) } catch (e: Exception) {}
                }

                val layout = FrameLayout(context)
                layout.setBackgroundColor(Color.parseColor("#F57C00"))

                // ── Vertikaler Container: Pfeil (optional) über dem Text,
                // beide zentriert — verhindert Überlappung mit "LOOK UP!".
                val content = LinearLayout(context)
                content.orientation = LinearLayout.VERTICAL
                content.gravity = Gravity.CENTER

                // Richtungspfeil (nur Main-Study VHA), eine "Zeile" über dem Text
                if (direction == "left" || direction == "right") {
                    val arrow = TextView(context)
                    arrow.text = if (direction == "left") "⬅️" else "➡️"
                    arrow.textSize = 72f
                    arrow.setTextColor(Color.WHITE)
                    arrow.gravity = Gravity.CENTER
                    val arrowParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                    arrowParams.bottomMargin = 16
                    content.addView(arrow, arrowParams)
                }

                val text = TextView(context)
                text.text = "⚠️ LOOK UP!"
                text.textSize = 48f
                text.setTextColor(Color.WHITE)
                text.gravity = Gravity.CENTER
                content.addView(text, LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ))

                layout.addView(content, FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    Gravity.CENTER
                ))

                val statusBarHeight = getStatusBarHeight(context)
                val navBarHeight    = getNavBarHeight(context)
                val screenHeight    = context.resources.displayMetrics.heightPixels
                val availableHeight = screenHeight - statusBarHeight - navBarHeight

                val params = WindowManager.LayoutParams(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    availableHeight,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                    PixelFormat.TRANSLUCENT
                )
                params.gravity = Gravity.TOP
                params.y = statusBarHeight

                wm.addView(layout, params)
                overlayView = layout
                android.util.Log.d("OverlayHelper", ">>> Overlay angezeigt!")

            } catch (e: Exception) {
                android.util.Log.e("OverlayHelper", ">>> Fehler: ${e.message}")
            }
        }
    }

    fun hideOverlay(context: Context) {
        Handler(Looper.getMainLooper()).post {
            try {
                val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                overlayView?.let {
                    wm.removeView(it)
                    overlayView = null
                    android.util.Log.d("OverlayHelper", ">>> Overlay versteckt")
                }
            } catch (e: Exception) {
                android.util.Log.e("OverlayHelper", ">>> Fehler beim Verstecken: ${e.message}")
            }
        }
    }

    // ── Partial edge overlay ("f" friendly alert) ────────────
    // Covers a fraction of the screen on the side the vehicle
    // approaches from. No text, no arrow, low intrusion.
    fun showEdgeOverlay(context: Context, direction: String, widthFraction: Float = 0.22f) {
        Handler(Looper.getMainLooper()).post {
            try {
                val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                windowManager = wm

                edgeOverlayView?.let {
                    try { wm.removeView(it) } catch (e: Exception) {}
                }

                val layout = FrameLayout(context)
                layout.setBackgroundColor(Color.parseColor("#CCFF3B30")) // translucent red

                val screenWidth  = context.resources.displayMetrics.widthPixels
                val overlayWidth = (screenWidth * widthFraction).toInt()

                val params = WindowManager.LayoutParams(
                    overlayWidth,
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                    PixelFormat.TRANSLUCENT
                )
                params.gravity = if (direction == "left") Gravity.START else Gravity.END

                wm.addView(layout, params)
                edgeOverlayView = layout
                android.util.Log.d("OverlayHelper", ">>> Edge-Overlay ($direction) angezeigt!")
            } catch (e: Exception) {
                android.util.Log.e("OverlayHelper", ">>> Fehler (edge): ${e.message}")
            }
        }
    }

    fun hideEdgeOverlay(context: Context) {
        Handler(Looper.getMainLooper()).post {
            try {
                val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                edgeOverlayView?.let {
                    wm.removeView(it)
                    edgeOverlayView = null
                    android.util.Log.d("OverlayHelper", ">>> Edge-Overlay versteckt")
                }
            } catch (e: Exception) {
                android.util.Log.e("OverlayHelper", ">>> Fehler (edge hide): ${e.message}")
            }
        }
    }

    private fun getStatusBarHeight(context: Context): Int {
        val resourceId = context.resources.getIdentifier(
            "status_bar_height", "dimen", "android"
        )
        return if (resourceId > 0) context.resources.getDimensionPixelSize(resourceId) else 0
    }

    private fun getNavBarHeight(context: Context): Int {
        val resourceId = context.resources.getIdentifier(
            "navigation_bar_height", "dimen", "android"
        )
        return if (resourceId > 0) context.resources.getDimensionPixelSize(resourceId) else 0
    }
}