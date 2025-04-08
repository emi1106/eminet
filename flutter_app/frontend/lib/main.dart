/*
 * Medical Diagnosis Chatbot Application
 * 
 * This app provides a multilingual interface for users to describe their 
 * symptoms and receive preliminary medical diagnoses.
 * 
 * Features:
 * - English and Romanian language support
 * - Backend API integration with Flask
 * - Persistent language preferences
 * - Animated text elements
 * - Dark mode interface
 */

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:animated_text_kit/animated_text_kit.dart';

/// Entry point for the application
/// Initializes necessary components and determines the initial language
void main() async {
  // Ensure Flutter binding is initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  // Load saved language preference or default to English
  SharedPreferences prefs = await SharedPreferences.getInstance();
  String languageCode = prefs.getString('language_code') ?? 'en';
  
  runApp(MyApp(languageCode: languageCode));
}

class MyApp extends StatefulWidget {
  final String languageCode;
  
  const MyApp({super.key, required this.languageCode});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _languageCode = 'en';
  
  @override
  void initState() {
    super.initState();
    _languageCode = widget.languageCode;
  }
  
  void _setLanguage(String languageCode) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('language_code', languageCode);
    setState(() {
      _languageCode = languageCode;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medical Diagnosis Chatbot',
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData.dark().copyWith(
        primaryColor: Colors.teal,
        scaffoldBackgroundColor: const Color(0xFF121212),
        colorScheme: ColorScheme.dark(
          primary: Colors.tealAccent,
          secondary: Colors.pinkAccent,
          surface: const Color(0xFF1E1E1E),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: const Color(0xFF1E1E1E),
          foregroundColor: Colors.tealAccent,
        ),
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.tealAccent),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.tealAccent, width: 2),
          ),
        ),
      ),
      locale: Locale(_languageCode),
      supportedLocales: const [
        Locale('en', ''), // English
        Locale('ro', ''), // Romanian
      ],
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      home: FutureBuilder(
        future: Future.delayed(Duration(seconds: 2)),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return SplashScreen(languageCode: _languageCode);
          } else {
            return MyHomePage(
              title: AppLocalizations.of(context)?.translate('app_title') ?? 'Medical Diagnosis Chatbot',
              setLanguage: _setLanguage,
              languageCode: _languageCode,
            );
          }
        },
      ),
    );
  }
}

class SplashScreen extends StatelessWidget {
  final String languageCode;
  
  const SplashScreen({super.key, required this.languageCode});
  
  @override
  Widget build(BuildContext context) {
    final welcomeText = languageCode == 'en' 
        ? 'Welcome to Medical Diagnosis Chatbot'
        : 'Bine ați venit la Chatbot-ul de Diagnostic Medical';
        
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.medical_services,
              size: 80,
              color: Colors.tealAccent,
            ),
            SizedBox(height: 20),
            AnimatedTextKit(
              animatedTexts: [
                TypewriterAnimatedText(
                  welcomeText,
                  textStyle: TextStyle(
                    fontSize: 22.0,
                    fontWeight: FontWeight.bold,
                    color: Colors.tealAccent,
                  ),
                  speed: Duration(milliseconds: 100),
                ),
              ],
              totalRepeatCount: 1,
            ),
            SizedBox(height: 30),
            CircularProgressIndicator(
              color: Colors.tealAccent,
            ),
          ],
        ),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final Function(String) setLanguage;
  final String languageCode;
  
  const MyHomePage({
    super.key, 
    required this.title, 
    required this.setLanguage,
    required this.languageCode,
  });

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with TickerProviderStateMixin {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, dynamic>> _messages = [];
  bool _isLoading = false;
  String _symptoms = "";
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    
    // Add initial bot greeting
    Future.delayed(const Duration(milliseconds: 500), () {
      setState(() {
        _messages.add({
          'sender': 'bot', 
          'text': AppLocalizations.of(context)?.translate('welcome_message') ?? 
                  'Hello! 👋 I\'m your medical assistant. Please describe your symptoms.',
          'timestamp': DateTime.now(),
        });
      });
    });
  }
  
  @override
  void dispose() {
    _tabController.dispose();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _sendMessage(String message) async {
    final timestamp = DateTime.now();
    setState(() {
      _messages.add({
        'sender': 'user', 
        'text': message,
        'timestamp': timestamp,
      });
      _isLoading = true;
    });

    final url = 'http://10.0.2.2:5000/predict';
    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "symptoms": _symptoms.isEmpty ? message : _symptoms,
          "follow_up": _symptoms.isEmpty ? "" : message,
          "language": widget.languageCode,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        String botEmoji = _getBotEmoji(data['diagnosis_message']);
        
        setState(() {
          _messages.add({
            'sender': 'bot', 
            'text': botEmoji + ' ' + data['diagnosis_message'],
            'timestamp': DateTime.now(),
          });
          
          if (data['follow_up_question'] != null && data['follow_up_question'].isNotEmpty) {
            _messages.add({
              'sender': 'bot', 
              'text': '🤔 ' + data['follow_up_question'],
              'timestamp': DateTime.now().add(Duration(milliseconds: 500)),
            });
            _symptoms = message;  // Store symptoms for follow-up
          } else {
            _symptoms = "";  // Reset symptoms after final diagnosis
          }
          _isLoading = false;
        });
      } else {
        setState(() {
          _messages.add({
            'sender': 'bot', 
            'text': '⚠️ ' + (AppLocalizations.of(context)?.translate('error_message') ?? 'Error: ') + response.body,
            'timestamp': DateTime.now(),
          });
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _messages.add({
          'sender': 'bot', 
          'text': '⚠️ ' + (AppLocalizations.of(context)?.translate('connection_error') ?? 'Connection error. Please try again later.'),
          'timestamp': DateTime.now(),
        });
        _isLoading = false;
      });
    }
  }
  
  String _getBotEmoji(String message) {
    if (message.toLowerCase().contains('emergency') || 
        message.toLowerCase().contains('urgent') ||
        message.toLowerCase().contains('urgență')) {
      return '🚨';
    } else if (message.toLowerCase().contains('recommend') || 
              message.toLowerCase().contains('suggest') ||
              message.toLowerCase().contains('recomand')) {
      return '💡';
    } else if (message.toLowerCase().contains('likely') || 
              message.toLowerCase().contains('probabil')) {
      return '🔍';
    } else {
      return '🩺';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        bottom: TabBar(
          controller: _tabController,
          tabs: [
            Tab(
              icon: Icon(Icons.chat),
              text: AppLocalizations.of(context)?.translate('chat_tab') ?? 'Chat',
            ),
            Tab(
              icon: Icon(Icons.settings),
              text: AppLocalizations.of(context)?.translate('settings_tab') ?? 'Settings',
            ),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildChatTab(),
          _buildSettingsTab(),
        ],
      ),
    );
  }
  
  Widget _buildChatTab() {
    return Column(
      children: <Widget>[
        Expanded(
          child: _messages.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.medical_services_outlined,
                        size: 48,
                        color: Colors.tealAccent.withOpacity(0.5),
                      ),
                      SizedBox(height: 16),
                      Text(
                        AppLocalizations.of(context)?.translate('start_conversation') ?? 
                        'Start a conversation with the medical assistant',
                        style: TextStyle(color: Colors.grey),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: EdgeInsets.all(12),
                  itemCount: _messages.length,
                  itemBuilder: (context, index) {
                    final message = _messages[index];
                    final isUser = message['sender'] == 'user';
                    
                    return AnimatedChatMessage(
                      message: message,
                      isUser: isUser,
                      index: index,
                    );
                  },
                ),
        ),
        if (_isLoading) 
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                SizedBox(width: 16),
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                SizedBox(width: 12),
                AnimatedTextKit(
                  animatedTexts: [
                    TypewriterAnimatedText(
                      AppLocalizations.of(context)?.translate('thinking') ?? 'Thinking...',
                      textStyle: TextStyle(
                        fontSize: 14.0,
                        color: Colors.grey,
                      ),
                      speed: Duration(milliseconds: 80),
                    ),
                  ],
                  repeatForever: true,
                ),
              ],
            ),
          ),
        Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            boxShadow: [
              BoxShadow(
                blurRadius: 8,
                color: Colors.black26,
                offset: Offset(0, -2),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 12.0),
          child: Row(
            children: <Widget>[
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration: InputDecoration(
                    labelText: AppLocalizations.of(context)?.translate('message_placeholder') ?? 'Enter your message',
                    prefixIcon: Icon(Icons.medical_information),
                    contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  ),
                  onSubmitted: (text) {
                    if (text.isNotEmpty) {
                      _sendMessage(text);
                      _controller.clear();
                    }
                  },
                ),
              ),
              SizedBox(width: 8),
              Material(
                color: Theme.of(context).colorScheme.primary,
                borderRadius: BorderRadius.circular(24),
                child: InkWell(
                  borderRadius: BorderRadius.circular(24),
                  onTap: () {
                    if (_controller.text.isNotEmpty) {
                      _sendMessage(_controller.text);
                      _controller.clear();
                    }
                  },
                  child: Container(
                    padding: EdgeInsets.all(12),
                    child: Icon(
                      Icons.send,
                      color: Colors.black,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
  
  Widget _buildSettingsTab() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            AppLocalizations.of(context)?.translate('language_settings') ?? 'Language Settings',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 20),
          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              children: [
                RadioListTile<String>(
                  title: Text('English 🇬🇧'),
                  value: 'en',
                  groupValue: widget.languageCode,
                  onChanged: (value) {
                    if (value != null) {
                      widget.setLanguage(value);
                    }
                  },
                  activeColor: Colors.tealAccent,
                ),
                Divider(height: 1),
                RadioListTile<String>(
                  title: Text('Română 🇷🇴'),
                  value: 'ro',
                  groupValue: widget.languageCode,
                  onChanged: (value) {
                    if (value != null) {
                      widget.setLanguage(value);
                    }
                  },
                  activeColor: Colors.tealAccent,
                ),
              ],
            ),
          ),
          SizedBox(height: 32),
          Text(
            AppLocalizations.of(context)?.translate('about_app') ?? 'About',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 16),
          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '🩺 ' + (AppLocalizations.of(context)?.translate('app_title') ?? 'Medical Diagnosis Chatbot'),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    AppLocalizations.of(context)?.translate('app_description') ?? 
                    'This application helps you get preliminary medical insights based on your symptoms. Always consult with a healthcare professional for proper diagnosis and treatment.',
                    style: TextStyle(
                      fontSize: 14,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text('Version 1.0.0'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class AnimatedChatMessage extends StatefulWidget {
  final Map<String, dynamic> message;
  final bool isUser;
  final int index;

  const AnimatedChatMessage({
    required this.message,
    required this.isUser,
    required this.index,
    Key? key,
  }) : super(key: key);

  @override
  State<AnimatedChatMessage> createState() => _AnimatedChatMessageState();
}

class _AnimatedChatMessageState extends State<AnimatedChatMessage> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SlideTransition(
      position: Tween<Offset>(
        begin: Offset(widget.isUser ? 1 : -1, 0),
        end: Offset.zero,
      ).animate(_animation),
      child: FadeTransition(
        opacity: _animation,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 4.0),
          child: Row(
            mainAxisAlignment: widget.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (!widget.isUser) 
                CircleAvatar(
                  backgroundColor: Colors.tealAccent,
                  child: Icon(Icons.medical_services, color: Colors.black),
                ),
              SizedBox(width: 8),
              Flexible(
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    color: widget.isUser 
                        ? Theme.of(context).colorScheme.secondary.withOpacity(0.8)
                        : Theme.of(context).colorScheme.surface.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.message['text'],
                        style: TextStyle(
                          fontSize: 16,
                          color: widget.isUser ? Colors.white : null,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        _formatTime(widget.message['timestamp']),
                        style: TextStyle(
                          fontSize: 10,
                          color: widget.isUser 
                              ? Colors.white70
                              : Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SizedBox(width: 8),
              if (widget.isUser) 
                CircleAvatar(
                  backgroundColor: Theme.of(context).colorScheme.secondary,
                  child: Icon(Icons.person, color: Colors.white),
                ),
            ],
          ),
        ),
      ),
    );
  }
  
  String _formatTime(DateTime dateTime) {
    return '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
  }
}

class AppLocalizations {
  final Locale locale;
  
  AppLocalizations(this.locale);
  
  static AppLocalizations? of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations);
  }
  
  static const LocalizationsDelegate<AppLocalizations> delegate = _AppLocalizationsDelegate();
  
  static final Map<String, Map<String, String>> _localizedValues = {
    'en': {
      'app_title': 'Medical Diagnosis Chatbot',
      'welcome_message': 'Hello! 👋 I\'m your medical assistant. Please describe your symptoms.',
      'error_message': 'Error: ',
      'connection_error': 'Connection error. Please try again later.',
      'chat_tab': 'Chat',
      'settings_tab': 'Settings',
      'start_conversation': 'Start a conversation with the medical assistant',
      'thinking': 'Thinking...',
      'message_placeholder': 'Enter your message',
      'language_settings': 'Language Settings',
      'about_app': 'About',
      'app_description': 'This application helps you get preliminary medical insights based on your symptoms. Always consult with a healthcare professional for proper diagnosis and treatment.',
    },
    'ro': {
      'app_title': 'Chatbot de Diagnostic Medical',
      'welcome_message': 'Bună! 👋 Sunt asistentul tău medical. Te rog să-mi descrii simptomele tale.',
      'error_message': 'Eroare: ',
      'connection_error': 'Eroare de conexiune. Vă rugăm să încercați mai târziu.',
      'chat_tab': 'Chat',
      'settings_tab': 'Setări',
      'start_conversation': 'Începe o conversație cu asistentul medical',
      'thinking': 'Se gândește...',
      'message_placeholder': 'Introduceți mesajul',
      'language_settings': 'Setări de Limbă',
      'about_app': 'Despre',
      'app_description': 'Această aplicație vă ajută să obțineți informații medicale preliminare pe baza simptomelor dvs. Consultați întotdeauna un profesionist în domeniul sănătății pentru diagnostic și tratament adecvat.',
    },
  };
  
  String? translate(String key) {
    return _localizedValues[locale.languageCode]?[key];
  }
}

class _AppLocalizationsDelegate extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();
  
  @override
  bool isSupported(Locale locale) {
    return ['en', 'ro'].contains(locale.languageCode);
  }
  
  @override
  Future<AppLocalizations> load(Locale locale) async {
    return AppLocalizations(locale);
  }
  
  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}
