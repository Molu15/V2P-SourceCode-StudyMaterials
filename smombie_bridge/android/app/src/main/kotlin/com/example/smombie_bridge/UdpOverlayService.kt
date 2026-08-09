package com.example.smombie_bridge

import android.app.Service
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.SoundPool
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import kotlinx.coroutines.*
import java.net.DatagramPacket
import java.net.DatagramSocket

class UdpOverlayService : Service() {

    private val job = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.IO + job)
    private var socket: DatagramSocket? = null

    private var isPlayingAlarm = false

    private val PULSE_MS = 400L
    private val PAUSE_MS = 200L

    // "f" edge highlight duration. Kept below TTC_EARLY_WARN - TTC_FULL_WARN
    // (= 5.0s - 2.5s = 2.5s real-time gap) so the edge overlay clears before
    // the VHA flash starts -> the two alert stages remain perceptually
    // distinct (relevant for habituation / alarm fatigue).
    private val EDGE_OVERLAY_DURATION_MS = 2000L

    // Lautstärke des Custom-Tons (0.0 = stumm, 1.0 = volle Stream-Lautstärke).
    // [tune] ggf. weiter anpassen nach Pilot-Test.
    private val ALERT_VOLUME = 0.4f

    private var soundPool: SoundPool? = null
    private var alertSoundId: Int = 0
    private var alertSoundLoaded = false

    override fun onCreate() {
        super.onCreate()
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ALARM)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()

        soundPool = SoundPool.Builder()
            .setMaxStreams(3)
            .setAudioAttributes(attrs)
            .build()

        soundPool?.setOnLoadCompleteListener { _, _, status ->
            alertSoundLoaded = (status == 0)
            android.util.Log.d("UdpOverlayService", ">>> Sound geladen: $alertSoundLoaded")
        }

        // res/raw/alert_tone.wav
        alertSoundId = soundPool!!.load(this, R.raw.alert_tone, 1)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        android.util.Log.d("UdpOverlayService", ">>> Service gestartet, höre auf Port 5008")
        scope.launch {
            try {
                socket = DatagramSocket(5008)
                android.util.Log.d("UdpOverlayService", ">>> Socket gebunden auf 5008")
                val buffer = ByteArray(1024)
                while (true) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket!!.receive(packet)
                    val msg = String(packet.data, 0, packet.length)
                    android.util.Log.d("UdpOverlayService", ">>> Paket empfangen: $msg")

                    if (msg.contains(":")) {
                        // ── Main-study protocol: "<code>:<direction>" ──
                        val parts     = msg.split(":")
                        val code      = parts.getOrElse(0) { "" }
                        val direction = parts.getOrNull(1)  // "left" | "right"

                        when (code) {
                            "f" -> {
                                // Friendly/adaptive early alert: partial
                                // edge highlight only — no sound/vibration/arrow.
                                if (direction == "left" || direction == "right") {
                                    android.util.Log.d("UdpOverlayService", ">>> Edge-Highlight ($direction)")
                                    OverlayHelper.showEdgeOverlay(applicationContext, direction)
                                    Handler(Looper.getMainLooper()).postDelayed({
                                        OverlayHelper.hideEdgeOverlay(applicationContext)
                                    }, EDGE_OVERLAY_DURATION_MS)
                                }
                            }
                            "vha" -> {
                                android.util.Log.d("UdpOverlayService", ">>> Voller Alarm ($direction)")
                                OverlayHelper.hideEdgeOverlay(applicationContext)
                                playFlashingOverlay(direction)
                                playCustomAlarmThreeTimes()
                                vibrateThreeTimes()
                            }
                            else -> handleLegacyCode(code)
                        }
                    } else {
                        // ── Pre-study protocol: undelimited modality codes ──
                        handleLegacyCode(msg)
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("UdpOverlayService", ">>> Fehler: ${e.message}")
            }
        }
        return START_STICKY
    }

    // ── Pre-study compatibility: 'v'/'a'/'h' substring codes ──
    // (sim_alarm.py sends e.g. "v", "vh", "vha" without delimiters)
    private fun handleLegacyCode(code: String) {
        if (code.contains("v")) {
            android.util.Log.d("UdpOverlayService", ">>> [legacy] Starte Overlay!")
            playFlashingOverlay(null)
        }
        if (code.contains("a")) {
            android.util.Log.d("UdpOverlayService", ">>> [legacy] Spiele Ton!")
            playAlarmThreeTimes()
        }
        if (code.contains("h")) {
            android.util.Log.d("UdpOverlayService", ">>> [legacy] Vibration!")
            vibrateThreeTimes()
        }
    }

    // ── 3x Vibration ────────────────────────────────────
    private fun vibrateThreeTimes() {
        val vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        // Schema: [Start-Pause, Puls, Pause, Puls, Pause, Puls]
        val pattern = longArrayOf(0, PULSE_MS, PAUSE_MS, PULSE_MS, PAUSE_MS, PULSE_MS)
        
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            vibrator.vibrate(pattern, -1)
        }
    }

    // ── 3x Ton ──────────────────────────────────────────
    private fun playAlarmThreeTimes() {
        if (isPlayingAlarm) return
        isPlayingAlarm = true

        try {
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            val handler = Handler(Looper.getMainLooper())
            val stream = AudioManager.STREAM_MUSIC

            // AudioFocus anfordern: Android pausiert Spotify/Podcast kurz (AUDIOFOCUS_GAIN_TRANSIENT),
            // spielt unseren Ton ab, und gibt danach den Focus wieder zurück.
            // Keine Lautstärke-Manipulation nötig – der Ton läuft auf der bestehenden Lautstärke.
            val focusResult = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                val focusRequest = android.media.AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(
                        android.media.AudioAttributes.Builder()
                            .setUsage(android.media.AudioAttributes.USAGE_ALARM)
                            .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                            .build()
                    )
                    .build()
                audioManager.requestAudioFocus(focusRequest)
                focusRequest // merken für späteren abandon
            } else {
                @Suppress("DEPRECATION")
                audioManager.requestAudioFocus(null, stream, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                null
            }

            // TONE_PROP_BEEP = einzelner, kurzer Beep ohne interne Wiederholung.
            // MAX_VOLUME bezieht sich auf den Stream-internen Pegel, nicht die System-Lautstärke.
            val toneGen = ToneGenerator(stream, ToneGenerator.MAX_VOLUME)

            for (i in 0..2) {
                handler.postDelayed({
                    toneGen.stopTone()
                    toneGen.startTone(ToneGenerator.TONE_PROP_BEEP, PULSE_MS.toInt())
                }, i * (PULSE_MS + PAUSE_MS))
            }

            // AudioFocus wieder freigeben → Spotify/Podcast läuft automatisch weiter
            val totalDuration = 3 * (PULSE_MS + PAUSE_MS) + 300
            handler.postDelayed({
                toneGen.stopTone()
                toneGen.release()
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    (focusResult as? android.media.AudioFocusRequest)?.let {
                        audioManager.abandonAudioFocusRequest(it)
                    }
                } else {
                    @Suppress("DEPRECATION")
                    audioManager.abandonAudioFocus(null)
                }
                isPlayingAlarm = false
            }, totalDuration)

        } catch (e: Exception) {
            android.util.Log.e("UdpOverlayService", ">>> Ton-Fehler: ${e.message}")
            isPlayingAlarm = false
        }
    }

    // ── 3x Custom-Ton (main-study "vha:<direction>") ─────
    private fun playCustomAlarmThreeTimes() {
        if (isPlayingAlarm) return
        isPlayingAlarm = true

        try {
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            val handler = Handler(Looper.getMainLooper())
            val stream = AudioManager.STREAM_MUSIC

            val focusResult = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                val focusRequest = android.media.AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ALARM)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                            .build()
                    )
                    .build()
                audioManager.requestAudioFocus(focusRequest)
                focusRequest
            } else {
                @Suppress("DEPRECATION")
                audioManager.requestAudioFocus(null, stream, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                null
            }

            if (!alertSoundLoaded) {
                android.util.Log.e("UdpOverlayService", ">>> alert_tone noch nicht geladen — überspringe")
            }

            for (i in 0..2) {
                handler.postDelayed({
                    soundPool?.play(alertSoundId, ALERT_VOLUME, ALERT_VOLUME, 1, 0, 1.0f)
                }, i * (PULSE_MS + PAUSE_MS))
            }

            val totalDuration = 3 * (PULSE_MS + PAUSE_MS) + 300
            handler.postDelayed({
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    (focusResult as? android.media.AudioFocusRequest)?.let {
                        audioManager.abandonAudioFocusRequest(it)
                    }
                } else {
                    @Suppress("DEPRECATION")
                    audioManager.abandonAudioFocus(null)
                }
                isPlayingAlarm = false
            }, totalDuration)

        } catch (e: Exception) {
            android.util.Log.e("UdpOverlayService", ">>> Custom-Ton-Fehler: ${e.message}")
            isPlayingAlarm = false
        }
    }

    // ── 3x Flash Overlay ────────────────────────────────
    // direction: "left" | "right" | null (null = no arrow, legacy behaviour)
    private fun playFlashingOverlay(direction: String? = null) {
        val handler = Handler(Looper.getMainLooper())

        for (i in 0..2) {
            // Overlay einblenden
            handler.postDelayed({
                OverlayHelper.showOverlay(applicationContext, direction)
            }, i * (PULSE_MS + PAUSE_MS))

            // Overlay ausblenden (nach der Puls-Dauer)
            handler.postDelayed({
                OverlayHelper.hideOverlay(applicationContext)
            }, (i * (PULSE_MS + PAUSE_MS)) + PULSE_MS)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        job.cancel()
        socket?.close()
        soundPool?.release()
        soundPool = null
        super.onDestroy()
    }
}