"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Loader2, FileText, Bot, User } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: { file: string; page: string }[];
};

type SourceRaw = any;
type Source = { file: string; page: string; raw?: SourceRaw };

export default function Home() {
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [typing, setTyping] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const normalizeSources = (rawSources: any): Source[] => {
    if (!Array.isArray(rawSources)) return [];
    return rawSources.map((s: any) => {
      if (typeof s === "string") return { file: s, page: "N/A" };
      const file = s.file || s.source || (s.metadata && (s.metadata.source || s.metadata.file)) || "unknown";
      const page = s.page || s.page_number || (s.metadata && s.metadata.page) || "N/A";
      return { file, page: String(page) };
    });
  };

  const askBackend = async () => {
    if (!question.trim()) return;

    const userMessage: Message = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);
    setTyping(true);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Lỗi: ${res.status} ${text}`);
      }

      const data = await res.json();
      const answer = data.answer ?? "Không có câu trả lời.";
      const sources = normalizeSources(data.sources);

      // Hiệu ứng gõ chữ
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: answer, sources },
        ]);
        setTyping(false);
      }, 300);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Lỗi: ${err.message}` },
      ]);
      setTyping(false);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askBackend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <Bot className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-xl font-bold text-gray-800">CV Assistant</h1>
            <p className="text-xs text-gray-500">Hỏi đáp dựa trên CV nội bộ</p>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto px-4 py-6 max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="text-center mt-20 text-gray-500">
              <Bot className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <p>Hãy đặt câu hỏi để bắt đầu trò chuyện!</p>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-3 ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                  )}

                  <div
                    className={`max-w-xl px-4 py-3 rounded-2xl ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-white text-gray-800 shadow-md"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>

                    {/* Nguồn */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 flex flex-wrap gap-2">
                        {msg.sources.map((src, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full"
                          >
                            <FileText className="w-3 h-3" />
                            {src.file} (tr. {src.page})
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {msg.role === "user" && (
                    <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                      <User className="w-5 h-5 text-white" />
                    </div>
                  )}
                </div>
              ))}

              {/* Typing indicator */}
              {typing && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  <div className="bg-white px-4 py-3 rounded-2xl shadow-md">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3 items-end">
            <textarea
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[56px] max-h-32"
              placeholder="Nhập câu hỏi..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button
              onClick={askBackend}
              disabled={loading || !question.trim()}
              className="mb-1 p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="text-center text-xs text-gray-500 mt-2">
            Nhấn <kbd className="px-1 py-0.5 bg-gray-200 rounded">Enter</kbd> để gửi,{" "}
            <kbd className="px-1 py-0.5 bg-gray-200 rounded">Shift + Enter</kbd> để xuống dòng
          </p>
        </div>
      </div>
    </div>
  );
}