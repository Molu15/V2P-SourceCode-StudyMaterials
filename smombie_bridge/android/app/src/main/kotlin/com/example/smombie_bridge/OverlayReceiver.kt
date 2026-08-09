package com.example.smombie_bridge

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class OverlayReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == "com.example.smombie_bridge.SHOW_OVERLAY") {
            OverlayHelper.showOverlay(context)
        }
    }
}