package com.example.smombie_bridge

import android.content.Intent
import id.flutter.flutter_background_service.BackgroundService

class SmombieBackgroundService : BackgroundService() {
    
    fun sendOverlayBroadcast() {
        val intent = Intent("com.example.smombie_bridge.SHOW_OVERLAY")
        intent.setPackage(packageName)
        sendBroadcast(intent)
    }
}