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
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:intl/intl.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'dart:math';

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
        colorScheme: const ColorScheme.dark(
          primary: Colors.tealAccent,
          secondary: Colors.pinkAccent,
          surface: Color(0xFF1E1E1E),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1E1E1E),
          foregroundColor: Colors.tealAccent,
        ),
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Colors.tealAccent),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Colors.tealAccent, width: 2),
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
        future: Future.delayed(const Duration(seconds: 2)),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return SplashScreen(languageCode: _languageCode);
          } else {
            return MyHomePage(
              title: AppLocalizations.of(context)?.translate('app_title') ??
                  'Medical Diagnosis Chatbot',
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
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF0F2027), Color(0xFF203A43), Color(0xFF2C5364)],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Hero(
                tag: 'appIcon',
                child: Container(
                  padding: EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.tealAccent.withOpacity(0.3),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: Icon(
                    Icons.medical_services,
                    size: 80,
                    color: Colors.tealAccent,
                  ),
                ),
              ),
              SizedBox(height: 30),
              AnimatedTextKit(
                animatedTexts: [
                  TypewriterAnimatedText(
                    welcomeText,
                    textStyle: const TextStyle(
                      fontSize: 24.0,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                    speed: const Duration(milliseconds: 80),
                  ),
                ],
                totalRepeatCount: 1,
              ),
              SizedBox(height: 40),
              CircularProgressIndicator(
                color: Colors.tealAccent,
                strokeWidth: 3,
              ),
            ],
          ),
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

class _MyHomePageState extends State<MyHomePage>
    with SingleTickerProviderStateMixin {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, dynamic>> _messages = [];
  bool _isLoading = false;
  String _symptoms = "";
  late TabController _tabController;
  final ScrollController _scrollController = ScrollController();

  // List to store chat histories
  List<Map<String, dynamic>> _chatHistories = []; // Current conversation ID
  String _currentConversationId = "";

  // Common symptoms based on language
  List<String> _commonSymptoms = [];
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _updateCommonSymptoms();
  }

  @override
  void didUpdateWidget(MyHomePage oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Update symptoms when language changes
    if (oldWidget.languageCode != widget.languageCode) {
      _updateCommonSymptoms();
    }
  }

  // Update common symptoms based on selected language
  void _updateCommonSymptoms() {
    if (widget.languageCode == "en") {
      setState(() {
        _commonSymptoms = [
          "Headache",
          "Fever",
          "Cough",
          "Fatigue",
          "Sore throat",
        ];
      });
    } else if (widget.languageCode == "ro") {
      setState(() {
        _commonSymptoms = [
          "Dureri de cap",
          "Febră",
          "Tuse",
          "Oboseală",
          "Durere în gât",
        ];
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _initialize();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    await _loadChatHistories();

    // Start a new conversation if no history exists
    if (_currentConversationId.isEmpty) {
      setState(() {
        _currentConversationId = _generateUniqueId();
        _messages.clear();
      });
    }

    // Add welcome message if this is a new conversation
    if (_messages.isEmpty) {
      setState(() {
        _messages.add({
          'sender': 'bot',
          'text': AppLocalizations.of(context)?.translate('welcome_message') ??
              'Hello! 👋 I\'m your medical assistant. Please describe your symptoms.',
          'timestamp': DateTime.now(),
        });
      });
    }
  }

  String _generateUniqueId() {
    return DateTime.now().millisecondsSinceEpoch.toString() +
        Random().nextInt(10000).toString();
  }

  void _startNewConversation() {
    if (_messages.isNotEmpty) {
      _saveCurrentConversation();
    }

    setState(() {
      _currentConversationId = _generateUniqueId();
      _messages.clear();

      _messages.add({
        'sender': 'bot',
        'text': AppLocalizations.of(context)?.translate('welcome_message') ??
            'Hello! 👋 I\'m your medical assistant. Please describe your symptoms.',
        'timestamp': DateTime.now(),
      });

      _controller.clear();
      _tabController.animateTo(0);
    });
  }

  void _loadConversation(String id) {
    final conversation = _chatHistories.firstWhere(
      (conv) => conv['id'] == id,
      orElse: () => {'id': '', 'messages': []},
    );

    if (conversation['id'] != '') {
      setState(() {
        _currentConversationId = conversation['id'];
        _messages.clear();
        _messages
            .addAll(List<Map<String, dynamic>>.from(conversation['messages']));
        _tabController.animateTo(0);
      });
    }
  }

  void _deleteConversation(String id) {
    setState(() {
      _chatHistories.removeWhere((conv) => conv['id'] == id);
    });

    _saveChatHistories();

    if (id == _currentConversationId) {
      _startNewConversation();
    }
  }

  Future<void> _exportConversation(String id, BuildContext context) async {
    final conversation = _chatHistories.firstWhere(
      (conv) => conv['id'] == id,
      orElse: () => {'id': '', 'messages': [], 'title': 'Chat'},
    );

    if (conversation['id'] == '') {
      return;
    }

    final String title = conversation['title'];
    final List<Map<String, dynamic>> messages =
        List<Map<String, dynamic>>.from(conversation['messages']);

    final StringBuffer exportText = StringBuffer();
    exportText.writeln('Medical Diagnosis Chat');
    exportText.writeln('Conversation: $title');
    exportText.writeln(
        'Date: ${DateFormat('yyyy-MM-dd HH:mm').format(DateTime.now())}');
    exportText.writeln('----------------------------------------');
    exportText.writeln();

    for (final message in messages) {
      final sender = message['sender'] == 'user' ? 'You' : 'Bot';
      final timestamp = message['timestamp'] != null
          ? DateFormat('HH:mm:ss').format(message['timestamp'])
          : '';
      exportText.writeln('[$timestamp] $sender:');
      exportText.writeln('${message['text']}');
      exportText.writeln();
    }

    try {
      final directory = await getApplicationDocumentsDirectory();
      final fileName =
          'chat_${DateFormat('yyyyMMdd_HHmmss').format(DateTime.now())}.txt';
      final path = '${directory.path}/$fileName';
      final File file = File(path);

      await file.writeAsString(exportText.toString());

      await Share.shareFiles(
        [path],
        text: 'Chat Export - $title',
        subject: 'Medical Diagnosis Conversation',
      );

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            AppLocalizations.of(context)?.translate('export_success') ??
                'Chat exported successfully',
          ),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            AppLocalizations.of(context)?.translate('export_error') ??
                'Error exporting chat: $e',
          ),
        ),
      );
    }
  }

  Future<void> _loadChatHistories() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? savedHistories = prefs.getString('chat_histories');
    if (savedHistories != null) {
      setState(() {
        _chatHistories =
            List<Map<String, dynamic>>.from(jsonDecode(savedHistories));
      });
    }
  }

  Future<void> _saveChatHistories() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('chat_histories', jsonEncode(_chatHistories));
  }

  void _saveCurrentConversation() {
    if (_messages.length >= 2) {
      final Map<String, dynamic> conversation = {
        'id': _currentConversationId,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'title': _getConversationTitle(),
        'messages': List<Map<String, dynamic>>.from(_messages)
      };

      int existingIndex =
          _chatHistories.indexWhere((c) => c['id'] == _currentConversationId);

      if (existingIndex != -1) {
        _chatHistories[existingIndex] = conversation;
      } else {
        _chatHistories.add(conversation);
      }

      _saveChatHistories();
    }
  }

  String _getConversationTitle() {
    final firstUserMessage = _messages.firstWhere(
      (message) => message['sender'] == 'user',
      orElse: () => {
        'text': 'Chat ${DateFormat('MMM d, yyyy HH:mm').format(DateTime.now())}'
      },
    );

    String title = firstUserMessage['text'] as String;
    if (title.length > 30) {
      title = '${title.substring(0, 27)}...';
    }
    return title;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'EmiNet',
          style: TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.add),
            tooltip:
                AppLocalizations.of(context)?.translate('new_conversation') ??
                    'New Conversation',
            onPressed: _startNewConversation,
          ),
          IconButton(
            icon: Icon(Icons.language),
            tooltip:
                AppLocalizations.of(context)?.translate('change_language') ??
                    'Change Language',
            onPressed: () {
              _showLanguageSelectionDialog();
            },
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          tabs: [
            Tab(
              text:
                  AppLocalizations.of(context)?.translate('chat_tab') ?? 'Chat',
              icon: Icon(Icons.chat),
            ),
            Tab(
              text: AppLocalizations.of(context)?.translate('history_tab') ??
                  'History',
              icon: Icon(Icons.history),
            ),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          // Chat tab content
          Column(
            children: [
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  padding: EdgeInsets.all(8.0),
                  itemCount: _messages.length,
                  itemBuilder: (context, index) {
                    return _buildMessage(_messages[index]);
                  },
                ),
              ),
              Divider(height: 1.0),
              _buildInputArea(),
            ],
          ),

          // History tab content
          _buildHistoryTab(),
        ],
      ),
    );
  }

  // Build the history tab content
  Widget _buildHistoryTab() {
    if (_chatHistories.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.history, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              AppLocalizations.of(context)?.translate('no_history') ??
                  'No conversation history',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
            SizedBox(height: 24),
            ElevatedButton.icon(
              icon: Icon(Icons.add_comment),
              label: Text(
                  AppLocalizations.of(context)?.translate('start_new_chat') ??
                      'Start a new chat'),
              onPressed: _startNewConversation,
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      itemCount: _chatHistories.length,
      padding: EdgeInsets.all(8.0),
      itemBuilder: (context, index) {
        final conversation = _chatHistories[index];
        final String id = conversation['id'];
        final List<dynamic> messages = conversation['messages'];

        // Generate a title for the conversation from the first message or timestamp
        String title = 'Chat';
        if (messages.isNotEmpty) {
          // Try to get the first user message as the title
          final userMessage = messages.firstWhere(
            (msg) => msg['sender'] == 'user',
            orElse: () => {'text': ''},
          );

          if (userMessage['text'] != '') {
            title = userMessage['text'].toString().length > 30
                ? '${userMessage['text'].toString().substring(0, 30)}...'
                : userMessage['text'].toString();
          } else {
            // Fallback to timestamp
            final DateTime timestamp =
                messages.first['timestamp'] ?? DateTime.now();
            title = 'Chat ${DateFormat('MMM d, HH:mm').format(timestamp)}';
          }
        }

        // Save the title for future use
        if (conversation['title'] == null ||
            conversation['title'].toString().isEmpty) {
          conversation['title'] = title;
          _saveChatHistories();
        } else {
          title = conversation['title'];
        }

        final bool isActive = id == _currentConversationId;
        final int messageCount = messages.length;
        final DateTime timestamp = messages.isNotEmpty
            ? (messages.last['timestamp'] ?? DateTime.now())
            : DateTime.now();

        return Card(
          elevation: isActive ? 3 : 1,
          color: isActive ? Theme.of(context).colorScheme.surfaceVariant : null,
          margin: EdgeInsets.symmetric(vertical: 4.0),
          child: ListTile(
            title: Text(
              title,
              style: TextStyle(
                fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
              ),
            ),
            subtitle: Text(
              '${DateFormat('MMM d, HH:mm').format(timestamp)} · $messageCount messages',
              style: TextStyle(fontSize: 12),
            ),
            leading: CircleAvatar(
              backgroundColor: isActive
                  ? Theme.of(context).colorScheme.primary
                  : Colors.grey,
              child: Icon(
                Icons.chat,
                color: Colors.white,
              ),
            ),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(
                  icon: Icon(Icons.ios_share),
                  tooltip:
                      AppLocalizations.of(context)?.translate('export_chat') ??
                          'Export chat',
                  onPressed: () => _exportConversation(id, context),
                ),
                IconButton(
                  icon: Icon(Icons.delete_outline),
                  tooltip:
                      AppLocalizations.of(context)?.translate('delete_chat') ??
                          'Delete chat',
                  onPressed: () {
                    showDialog(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: Text(AppLocalizations.of(context)
                                ?.translate('confirm_delete') ??
                            'Confirm delete'),
                        content: Text(AppLocalizations.of(context)
                                ?.translate('delete_chat_confirmation') ??
                            'Are you sure you want to delete this conversation?'),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.of(context).pop(),
                            child: Text(AppLocalizations.of(context)
                                    ?.translate('cancel') ??
                                'Cancel'),
                          ),
                          TextButton(
                            onPressed: () {
                              Navigator.of(context).pop();
                              _deleteConversation(id);
                            },
                            style: TextButton.styleFrom(
                              foregroundColor: Colors.red,
                            ),
                            child: Text(AppLocalizations.of(context)
                                    ?.translate('delete') ??
                                'Delete'),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ],
            ),
            onTap: () => _loadConversation(id),
          ),
        );
      },
    );
  }

  Widget _buildMessage(Map<String, dynamic> message) {
    final isUser = message['sender'] == 'user';
    return AnimatedChatMessage(
      message: message,
      isUser: isUser,
      index: _messages.indexOf(message),
    );
  }

  Widget _buildInputArea() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Suggestion chips for common symptoms
        if (_messages.length <=
            2) // Only show suggestions at the beginning of conversations
          Container(
            height: 50,
            margin: EdgeInsets.only(bottom: 8),
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: EdgeInsets.symmetric(horizontal: 8),
              itemCount: _commonSymptoms.length,
              itemBuilder: (context, index) {
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4.0),
                  child: ActionChip(
                    avatar: Icon(
                      Icons.medical_services,
                      size: 16,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                    label: Text(_commonSymptoms[index]),
                    elevation: 3,
                    backgroundColor: Theme.of(context).colorScheme.surface,
                    shape: StadiumBorder(
                      side: BorderSide(
                        color: Theme.of(context)
                            .colorScheme
                            .primary
                            .withOpacity(0.5),
                      ),
                    ),
                    onPressed: () {
                      setState(() {
                        _controller.text = _commonSymptoms[index];
                        _controller.selection = TextSelection.fromPosition(
                          TextPosition(offset: _controller.text.length),
                        );
                      });
                    },
                  ),
                );
              },
            ),
          ),
        Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            boxShadow: const [
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
                    labelText: AppLocalizations.of(context)
                            ?.translate('message_placeholder') ??
                        'Enter your message',
                    prefixIcon: const Icon(Icons.medical_information),
                    filled: true,
                    fillColor: Color(0xFF2A2A2A),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide.none,
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(
                        color: Colors.tealAccent,
                        width: 2,
                      ),
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                  ),
                  onSubmitted: (text) {
                    if (text.isNotEmpty) {
                      _sendMessage(text: text);
                      _controller.clear();
                    }
                  },
                ),
              ),
              const SizedBox(width: 8),
              Material(
                color: Theme.of(context).colorScheme.primary,
                borderRadius: BorderRadius.circular(24),
                child: InkWell(
                  borderRadius: BorderRadius.circular(24),
                  onTap: () {
                    if (_controller.text.isNotEmpty) {
                      _sendMessage(text: _controller.text);
                      _controller.clear();
                    }
                  },
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    child: const Icon(Icons.send, color: Colors.black),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _showLanguageSelectionDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(
            AppLocalizations.of(context)?.translate('language_settings') ??
                'Language Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<String>(
              title: const Text('English 🇬🇧'),
              value: 'en',
              groupValue: widget.languageCode,
              onChanged: (value) {
                if (value != null) {
                  widget.setLanguage(value);
                  Navigator.of(context).pop();
                }
              },
              activeColor: Colors.tealAccent,
            ),
            RadioListTile<String>(
              title: const Text('Română 🇷🇴'),
              value: 'ro',
              groupValue: widget.languageCode,
              onChanged: (value) {
                if (value != null) {
                  widget.setLanguage(value);
                  Navigator.of(context).pop();
                }
              },
              activeColor: Colors.tealAccent,
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _sendMessage({required String text}) async {
    final timestamp = DateTime.now();
    setState(() {
      _messages.add({'sender': 'user', 'text': text, 'timestamp': timestamp});
      _isLoading = true;
    });

    const url = 'http://10.0.2.2:5000/predict';
    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "symptoms": _symptoms.isEmpty ? text : _symptoms,
          "follow_up": _symptoms.isEmpty ? "" : text,
          "language": widget.languageCode,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        String botEmoji = _getBotEmoji(data['diagnosis_message']);

        setState(() {
          _messages.add({
            'sender': 'bot',
            'text': '$botEmoji ' + data['diagnosis_message'],
            'timestamp': DateTime.now(),
          });

          if (data['follow_up_question'] != null &&
              data['follow_up_question'].isNotEmpty) {
            _messages.add({
              'sender': 'bot',
              'text': '🤔 ' + data['follow_up_question'],
              'timestamp': DateTime.now().add(
                const Duration(milliseconds: 500),
              ),
            });
            _symptoms = text; // Store symptoms for follow-up
          } else {
            _symptoms = ""; // Reset symptoms after final diagnosis
          }
          _isLoading = false;
        });
      } else {
        setState(() {
          _messages.add({
            'sender': 'bot',
            'text':
                '⚠️ ${AppLocalizations.of(context)?.translate('error_message') ?? 'Error: '}${response.body}',
            'timestamp': DateTime.now(),
          });
          _isLoading = false;
        });
      }

      _saveCurrentConversation();
    } catch (e) {
      setState(() {
        _messages.add({
          'sender': 'bot',
          'text':
              '⚠️ ${AppLocalizations.of(context)?.translate('connection_error') ?? 'Connection error. Please try again later.'}',
          'timestamp': DateTime.now(),
        });
        _isLoading = false;
      });

      _saveCurrentConversation();
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

class _AnimatedChatMessageState extends State<AnimatedChatMessage>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
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
          padding: const EdgeInsets.symmetric(vertical: 6.0),
          child: Row(
            mainAxisAlignment:
                widget.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              if (!widget.isUser)
                Container(
                  margin: EdgeInsets.only(right: 8),
                  child: Stack(
                    children: [
                      CircleAvatar(
                        backgroundColor: Colors.tealAccent,
                        child: Icon(
                          Icons.medical_services,
                          color: Colors.black,
                        ),
                      ),
                      Positioned(
                        right: 0,
                        bottom: 0,
                        child: Container(
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            color: Colors.green,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: Theme.of(context).scaffoldBackgroundColor,
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              Flexible(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                  decoration: BoxDecoration(
                    color: widget.isUser
                        ? Theme.of(context).colorScheme.secondary
                        : Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(20).copyWith(
                      bottomLeft: widget.isUser
                          ? Radius.circular(20)
                          : Radius.circular(0),
                      bottomRight: !widget.isUser
                          ? Radius.circular(20)
                          : Radius.circular(0),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black12,
                        blurRadius: 5,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SelectableText(
                        widget.message['text'],
                        style: TextStyle(
                          fontSize: 16,
                          color: widget.isUser ? Colors.white : null,
                          height: 1.3,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        _formatTime(widget.message['timestamp']),
                        style: TextStyle(
                          fontSize: 10,
                          color: widget.isUser ? Colors.white70 : Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              if (widget.isUser)
                Container(
                  margin: EdgeInsets.only(left: 8),
                  child: CircleAvatar(
                    backgroundColor: Theme.of(context).colorScheme.secondary,
                    child: Icon(Icons.person, color: Colors.white),
                  ),
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

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  static final Map<String, Map<String, String>> _localizedValues = {
    'en': {
      'app_title': 'Medical Diagnosis Chatbot',
      'welcome_message':
          'Hello! 👋 I\'m your medical assistant. Please describe your symptoms.',
      'error_message': 'Error: ',
      'connection_error': 'Connection error. Please try again later.',
      'chat_tab': 'Chat',
      'settings_tab': 'Settings',
      'history_tab': 'History',
      'start_conversation': 'Start a conversation with the medical assistant',
      'thinking': 'Thinking...',
      'message_placeholder': 'Enter your message',
      'language_settings': 'Language Settings',
      'about_app': 'About',
      'app_description':
          'This application helps you get preliminary medical insights based on your symptoms. Always consult with a healthcare professional for proper diagnosis and treatment.',
      'clear_chat_title': 'Clear chat history?',
      'clear_chat_message':
          'This will remove all messages from this conversation.',
      'cancel': 'Cancel',
      'clear': 'Clear',
      'voice_input': 'Voice input',
      'no_history': 'No conversation history yet',
      'export': 'Export',
      'session_summary': 'Session Summary',
      'no_messages': 'No messages in this session.',
      'load': 'Load',
      'delete': 'Delete',
      'export_success': 'Chat exported successfully',
      'export_error': 'Error exporting chat',
      'new_conversation': 'New Conversation',
      'change_language': 'Change Language',
      'export_chat': 'Export chat',
      'delete_chat': 'Delete chat',
      'confirm_delete': 'Confirm delete',
      'delete_chat_confirmation':
          'Are you sure you want to delete this conversation?',
      'start_new_chat': 'Start a new chat',
    },
    'ro': {
      'app_title': 'Chatbot de Diagnostic Medical',
      'welcome_message':
          'Bună! 👋 Sunt asistentul tău medical. Te rog să-mi descrii simptomele tale.',
      'error_message': 'Eroare: ',
      'connection_error':
          'Eroare de conexiune. Vă rugăm să încercați mai târziu.',
      'chat_tab': 'Chat',
      'settings_tab': 'Setări',
      'history_tab': 'Istoric',
      'start_conversation': 'Începe o conversație cu asistentul medical',
      'thinking': 'Se gândește...',
      'message_placeholder': 'Introduceți mesajul',
      'language_settings': 'Setări de Limbă',
      'about_app': 'Despre',
      'app_description':
          'Această aplicație vă ajută să obțineți informații medicale preliminare pe baza simptomelor dvs. Consultați întotdeauna un profesionist în domeniul sănătății pentru diagnostic și tratament adecvat.',
      'clear_chat_title': 'Ștergeți istoricul conversației?',
      'clear_chat_message':
          'Acest lucru va elimina toate mesajele din această conversație.',
      'cancel': 'Anulare',
      'clear': 'Șterge',
      'voice_input': 'Intrare vocală',
      'no_history': 'Încă nu există istoric de conversație',
      'export': 'Exportă',
      'session_summary': 'Rezumatul sesiunii',
      'no_messages': 'Niciun mesaj în această sesiune.',
      'load': 'Încarcă',
      'delete': 'Șterge',
      'export_success': 'Chat exportat cu succes',
      'export_error': 'Eroare la exportul chatului',
      'new_conversation': 'Conversație nouă',
      'change_language': 'Schimbă limba',
      'export_chat': 'Exportă conversația',
      'delete_chat': 'Șterge conversația',
      'confirm_delete': 'Confirmă ștergerea',
      'delete_chat_confirmation':
          'Sigur doriți să ștergeți această conversație?',
      'start_new_chat': 'Începe un chat nou',
    },
  };

  String? translate(String key) {
    return _localizedValues[locale.languageCode]?[key];
  }
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
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
