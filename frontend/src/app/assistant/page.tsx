'use client';

import { formatTime, useHydrated } from '@/hooks/useHydration';
import { mlApi } from '@/lib/api';
import { clsx } from 'clsx';
import { Bot, Loader2, RefreshCw, Send, Sparkles, User } from 'lucide-react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number; // Use timestamp number to avoid hydration issues
}

// Memoized message component for better performance
const ChatMessage = memo(function ChatMessage({
  message,
  isHydrated
}: {
  message: Message;
  isHydrated: boolean;
}) {
  return (
    <div
      className={clsx(
        'flex items-start gap-2 sm:gap-3',
        message.role === 'user' ? 'flex-row-reverse' : ''
      )}
    >
      <div
        className={clsx(
          'h-7 w-7 sm:h-9 sm:w-9 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm',
          message.role === 'user'
            ? 'bg-blue-500'
            : 'bg-gradient-to-br from-purple-500 to-blue-500'
        )}
      >
        {message.role === 'user' ? (
          <User className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
        ) : (
          <Bot className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
        )}
      </div>
      <div
        className={clsx(
          'max-w-[80%] sm:max-w-[75%] rounded-2xl px-3 sm:px-4 py-2 sm:py-3 shadow-sm',
          message.role === 'user'
            ? 'bg-blue-500 text-white'
            : 'bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-white'
        )}
      >
        <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
        <p
          className={clsx(
            'text-xs mt-1 sm:mt-2',
            message.role === 'user' ? 'text-blue-200' : 'text-slate-400'
          )}
        >
          {isHydrated ? formatTime(message.timestamp) : '--:--'}
        </p>
      </div>
    </div>
  );
});

export default function AssistantPage() {
  const isHydrated = useHydrated();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: 'assistant',
      content:
        "Hello! I'm your AI restaurant assistant powered by Groq. I can help you with menu information, recommendations, dietary restrictions, and operational questions. What would you like to know?",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Check if ML service is available
  useEffect(() => {
    let isMounted = true;

    const checkConnection = async () => {
      try {
        const res = await mlApi.getHealth();
        if (isMounted) {
          const isLoaded = res.data?.models?.assistant === 'loaded';
          setIsConnected(isLoaded);
          if (isLoaded) {
            console.log('AI Assistant connected successfully');
          }
        }
      } catch (err) {
        if (isMounted) {
          console.error('ML health check failed:', err);
          setIsConnected(false);
        }
      }
    };

    // Initial check with slight delay to ensure API is ready
    const initialTimeout = setTimeout(checkConnection, 500);

    // Retry every 3 seconds until connected
    const interval = setInterval(() => {
      checkConnection();
    }, 3000);

    return () => {
      isMounted = false;
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, []);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await mlApi.chat({
        message: userMessage.content,
        use_rag: true
      });

      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.response || 'I received your message but could not generate a response.',
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorContent = error.response?.data?.detail ||
        'Sorry, I encountered an error connecting to the AI service. Please make sure the backend is running.';

      const errorMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: errorContent,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading]);

  const resetConversation = useCallback(async () => {
    try {
      await mlApi.chat({ message: '', use_rag: false, reset_conversation: true } as any);
    } catch (e) {
      // Ignore reset errors
    }

    setMessages([
      {
        id: Date.now(),
        role: 'assistant',
        content: "Conversation reset. How can I help you today?",
        timestamp: Date.now(),
      },
    ]);
  }, []);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)] sm:h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3 sm:mb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Sparkles className="h-5 w-5 sm:h-6 sm:w-6 text-purple-500" />
            AI Assistant
          </h1>
          <p className="text-sm text-slate-500">
            Ask me anything about the menu, recommendations, or operations
          </p>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Connection Status */}
          <div className={clsx(
            'flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full text-xs font-medium',
            isConnected === null && 'bg-slate-100 text-slate-600',
            isConnected === true && 'bg-green-100 text-green-700',
            isConnected === false && 'bg-red-100 text-red-700'
          )}>
            <span className={clsx(
              'w-2 h-2 rounded-full',
              isConnected === null && 'bg-slate-400 animate-pulse',
              isConnected === true && 'bg-green-500',
              isConnected === false && 'bg-red-500'
            )} />
            <span className="hidden sm:inline">
              {isConnected === null && 'Connecting...'}
              {isConnected === true && 'AI Connected'}
              {isConnected === false && 'AI Offline'}
            </span>
          </div>

          <button
            onClick={resetConversation}
            className="flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300"
          >
            <RefreshCw className="h-4 w-4" />
            <span className="hidden sm:inline">New Chat</span>
          </button>
        </div>
      </div>

      {/* Chat Container */}
      <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 flex flex-col overflow-hidden shadow-sm">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              isHydrated={isHydrated}
            />
          ))}

          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="h-9 w-9 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center shadow-sm">
                <Bot className="h-5 w-5 text-white" />
              </div>
              <div className="bg-slate-100 dark:bg-slate-700 rounded-2xl px-4 py-3 shadow-sm">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-purple-500" />
                  <span className="text-sm text-slate-500">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestions */}
        <div className="px-4 py-3 border-t border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
          <p className="text-xs text-slate-500 mb-2">Quick suggestions:</p>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {[
              'What vegetarian options do you have?',
              'Recommend something spicy',
              "What's popular today?",
              'Any gluten-free dishes?',
              'What desserts do you have?',
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInput(suggestion)}
                className="px-3 py-1.5 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-full text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-600 whitespace-nowrap transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>

        {/* Input */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isConnected === false ? "AI service offline..." : "Ask me anything about the menu..."}
              className="flex-1 px-4 py-3 bg-slate-100 dark:bg-slate-700 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white placeholder:text-slate-400"
              disabled={isLoading || isConnected === false}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading || isConnected === false}
              className="p-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
