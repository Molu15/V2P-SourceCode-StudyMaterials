package com.example.smombie_bridge

import android.content.Intent
import android.media.AudioManager
import android.media.ToneGenerator
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // UdpOverlayService starten
        val intent = Intent(this, UdpOverlayService::class.java)
        startService(intent)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "smombie/overlay")
            .setMethodCallHandler { call, result ->
                result.notImplemented()
            }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "smombie/audio")
            .setMethodCallHandler { call, result ->
                if (call.method == "playAlarm") {
                    try {
                        val toneGen = ToneGenerator(AudioManager.STREAM_ALARM, 100)
                        toneGen.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000)
                        result.success(null)
                    } catch (e: Exception) {
                        result.error("ERROR", e.message, null)
                    }
                } else {
                    result.notImplemented()
                }
            }
    }
}