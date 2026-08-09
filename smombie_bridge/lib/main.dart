import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:udp/udp.dart' as udp_lib;
import 'package:vibration/vibration.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_background_service_android/flutter_background_service_android.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ─────────────────────────────────────────────
// BACKGROUND SERVICE SETUP
// ─────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializeService();
  runApp(const SmombieApp());
}

Future<void> initializeService() async {
  final service = FlutterBackgroundService();
  const AndroidNotificationChannel channel = AndroidNotificationChannel(
    'smombie_v15', 'Critical Alarms',
    importance: Importance.max,
    playSound: false,
  );
  final notifications = FlutterLocalNotificationsPlugin();
  await notifications.initialize(
    settings: const InitializationSettings(
      android: AndroidInitializationSettings('@drawable/ic_notification'),
    ),
  );
  await notifications
      .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
      ?.createNotificationChannel(channel);

  await service.configure(
    androidConfiguration: AndroidConfiguration(
      onStart: onStart,
      autoStart: false,
      isForegroundMode: true,
      notificationChannelId: 'smombie_v15',
      initialNotificationTitle: 'Bridge Active',
      initialNotificationContent: 'Monitoring Sensors...',
      foregroundServiceTypes: [AndroidForegroundType.dataSync],
    ),
    iosConfiguration: IosConfiguration(),
  );
}

// ─────────────────────────────────────────────
// BACKGROUND SERVICE LOGIC
// ─────────────────────────────────────────────
@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {

  // Broadcast (255.255.255.255) funktioniert auf diesem Hotspot nicht
  // (Hairpin-Problem: Android gibt selbst gesendete Broadcasts nicht an
  // die eigene AP-Schnittstelle zurück). Daher Unicast an eine in der
  // UI hinterlegte Laptop-IP, persistiert via shared_preferences.
  final RawDatagramSocket sender =
      await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);

  final prefs = await SharedPreferences.getInstance();
  final String laptopIP = prefs.getString('laptop_ip') ?? "10.235.222.106";
  print("[Bridge] Sende Sensordaten an: $laptopIP:5006");
  bool _loggedFirstSend = false;

  final receiver = await udp_lib.UDP.bind(
    udp_lib.Endpoint.any(port: const udp_lib.Port(5007)),
  );

  service.on('stopService').listen((event) {
    sender.close();
    receiver.close();
    service.stopSelf();
  });

  receiver.asStream().listen((datagram) async {
    if (datagram == null) return;
    final code = String.fromCharCodes(datagram.data);

    if (code.contains('h')) {
      // Vibration.vibrate(duration: 500);
    }

    if (code.contains('v')) {
      // service.invoke("showOverlay", {});
    }
  });

  double lastX = 0.0;
  double lastY = 0.0;
  double lastZ = 0.0;
  accelerometerEventStream().listen((e) {
    lastX = e.x;
    lastY = e.y;
    lastZ = e.z;
  });
  gyroscopeEventStream().listen((e) {
    final msg = jsonEncode({
      "yaw":      e.z,
      "accel_x":  lastX,
      "accel_y":  lastY,
      "accel_z":  lastZ,
      "timestamp": DateTime.now().millisecondsSinceEpoch,
    });
    try {
      final bytesSent = sender.send(
        msg.codeUnits,
        InternetAddress(laptopIP),
        5006,
      );
      if (!_loggedFirstSend) {
        _loggedFirstSend = true;
        print("[Bridge] Erstes Paket gesendet ($bytesSent bytes) an $laptopIP:5006");
      }
    } catch (e) {
      print("[Bridge] FEHLER beim Senden: $e");
    }
  });
}

// ─────────────────────────────────────────────
// HAUPT-APP UI
// ─────────────────────────────────────────────
class SmombieApp extends StatefulWidget {
  const SmombieApp({super.key});
  @override
  State<SmombieApp> createState() => _SmombieAppState();
}

class _SmombieAppState extends State<SmombieApp> {
  bool _isRunning = false;
  final TextEditingController _ipController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Korrekten Status beim App-Start laden
    _checkServiceStatus();
    _loadSavedIP();
  }

  Future<void> _checkServiceStatus() async {
    final running = await FlutterBackgroundService().isRunning();
    if (mounted) setState(() => _isRunning = running);
  }

  Future<void> _loadSavedIP() async {
    final prefs = await SharedPreferences.getInstance();
    _ipController.text = prefs.getString('laptop_ip') ?? "10.235.222.106";
    if (mounted) setState(() {});
  }

  Future<void> _saveIP(String ip) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('laptop_ip', ip.trim());
  }

  void _toggle() async {
    final service = FlutterBackgroundService();
    if (await service.isRunning()) {
      service.invoke("stopService");
      setState(() => _isRunning = false);
    } else {
      await _saveIP(_ipController.text);
      await Permission.notification.request();
      if (!await FlutterOverlayWindow.isPermissionGranted()) {
        await FlutterOverlayWindow.requestPermission();
      }
      await service.startService();
      setState(() => _isRunning = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: Scaffold(
        backgroundColor: const Color(0xFF0A0A0A),
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Column(
              children: [
                const Spacer(flex: 2),

                // Status-Anzeige oben
                AnimatedContainer(
                  duration: const Duration(milliseconds: 400),
                  width: 180,
                  height: 180,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isRunning
                        ? const Color(0xFF1A3A1A)
                        : const Color(0xFF1A1A1A),
                    border: Border.all(
                      color: _isRunning
                          ? const Color(0xFF4CAF50)
                          : const Color(0xFF444444),
                      width: 2,
                    ),
                    boxShadow: _isRunning
                        ? [BoxShadow(
                            color: const Color(0xFF4CAF50).withOpacity(0.3),
                            blurRadius: 40,
                            spreadRadius: 10,
                          )]
                        : [],
                  ),
                  child: Icon(
                    _isRunning ? Icons.sensors : Icons.sensors_off,
                    color: _isRunning
                        ? const Color(0xFF4CAF50)
                        : const Color(0xFF666666),
                    size: 80,
                  ),
                ),

                const SizedBox(height: 32),

                // Status-Text
                Text(
                  _isRunning ? 'ACTIVE' : 'INACTIVE',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 4,
                    color: _isRunning
                        ? const Color(0xFF4CAF50)
                        : const Color(0xFF666666),
                  ),
                ),

                const SizedBox(height: 8),

                Text(
                  _isRunning
                      ? 'Sensors transmitting'
                      : 'Bridge not started',
                  style: const TextStyle(
                    fontSize: 14,
                    color: Color(0xFF888888),
                  ),
                ),

                const Spacer(flex: 3),

                // Laptop-IP — editierbar solange Bridge gestoppt ist
                TextField(
                  controller: _ipController,
                  enabled: !_isRunning,
                  style: const TextStyle(color: Colors.white),
                  keyboardType: TextInputType.text,
                  decoration: InputDecoration(
                    labelText: 'Laptop-IP',
                    hintText: 'e.g., 10.235.222.106',
                    labelStyle: const TextStyle(color: Color(0xFF888888)),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onSubmitted: _saveIP,
                  onEditingComplete: () => _saveIP(_ipController.text),
                ),
                const SizedBox(height: 4),
                Text(
                  _isRunning
                      ? 'IP can only be changed after stopping the bridge'
                      : 'Will be used on the next bridge start',
                  style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
                ),

                const SizedBox(height: 16),

                // Start/Stop Button
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 400),
                    child: ElevatedButton(
                      onPressed: _toggle,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isRunning
                            ? const Color(0xFF2A1A1A)
                            : const Color(0xFFF57C00),
                        foregroundColor: _isRunning
                            ? const Color(0xFFFF5252)
                            : Colors.white,
                        side: BorderSide(
                          color: _isRunning
                              ? const Color(0xFFFF5252)
                              : Colors.transparent,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        elevation: 0,
                      ),
                      child: Text(
                        _isRunning ? 'STOP BRIDGE' : 'START BRIDGE',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 2,
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Info-Text unten
                Text(
                  _isRunning
                      ? 'Running in the background'
                      : 'Tap to start',
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFF555555),
                  ),
                ),

                const Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}