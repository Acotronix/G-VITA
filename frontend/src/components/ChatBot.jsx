import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Maximize2,
  Minimize2,
} from "lucide-react";

import "../styles/chatbot.css";

// Backend API
const CHAT_API_URL = "http://localhost:5000/api/chat";

const INITIAL_MESSAGES = [
  {
    id: 1,
    sender: "bot",
    text: "Hi! I'm your VITA Yoga Assistant 🧘",
  },
  {
    id: 2,
    sender: "bot",
    text: "Ask me about yoga poses, breathing, alignment, or your practice.",
  },
];

function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Automatically scroll to the latest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isTyping]);

  // Focus the input when the chat opens.
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  const handleClose = useCallback(() => {
    setIsOpen(false);
    setIsMaximized(false);
  }, []);

  // Send the question to the local AI backend.
  const sendMessage = useCallback(async () => {
    const question = input.trim();

    if (!question || isTyping) return;

    const userMessage = {
      id: `${Date.now()}-user`,
      sender: "user",
      text: question,
    };

    setMessages((previous) => [...previous, userMessage]);
    setInput("");
    setError("");
    setIsTyping(true);

    try {
      const response = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `The AI server returned an error (${response.status}).`
        );
      }

      const data = await response.json();

      // Supports the current backend response format: { answer: "..." }
      const answer =
        typeof data.answer === "string" ? data.answer.trim() : "";

      if (!answer) {
        throw new Error("The AI returned an empty response.");
      }

      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-bot`,
          sender: "bot",
          text: answer,
        },
      ]);
    } catch (err) {
      console.error("VITA chatbot error:", err);

      const errorMessage =
        "I couldn't connect to the local AI server. Please check that Ollama and your VITA backend are running, then try again.";

      setError(err.message || errorMessage);

      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-error`,
          sender: "bot",
          text: errorMessage,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping]);

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* CHAT WINDOW */}
      {isOpen && (
        <div
          className={`chatbot-window${
            isMaximized ? " chatbot-window--max" : ""
          }`}
          role="dialog"
          aria-label="VITA Yoga Assistant"
          aria-modal="false"
        >
          {/* HEADER */}
          <div className="chatbot-header">
            <div className="chatbot-header-info">
              <div className="chatbot-avatar" aria-hidden="true">
                <Bot size={20} />
              </div>

              <div>
                <h3>VITA Assistant</h3>

                <span>
                  <span className="online-dot" aria-hidden="true" />
                  Local AI Assistant
                </span>
              </div>
            </div>

            <div className="chatbot-header-actions">
              <button
                type="button"
                className="chatbot-header-btn"
                onClick={() => setIsMaximized((previous) => !previous)}
                aria-label={
                  isMaximized ? "Minimize chat" : "Maximize chat"
                }
                title={isMaximized ? "Minimize" : "Maximize"}
              >
                {isMaximized ? (
                  <Minimize2 size={16} />
                ) : (
                  <Maximize2 size={16} />
                )}
              </button>

              <button
                type="button"
                className="chatbot-header-btn"
                onClick={handleClose}
                aria-label="Close chat"
                title="Close"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* MESSAGES */}
          <div
            className="chatbot-messages"
            aria-live="polite"
            aria-label="Chat messages"
            aria-busy={isTyping}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                className={`chat-message ${
                  message.sender === "user"
                    ? "user-message"
                    : "bot-message"
                }`}
              >
                {message.sender !== "user" && (
                  <div
                    className="message-avatar bot-message-avatar"
                    aria-hidden="true"
                  >
                    <Bot size={15} />
                  </div>
                )}

                <div className="message-bubble">
                  {message.sender === "user" ? (
                    message.text
                  ) : (
                    <ReactMarkdown>
                      {message.text}
                    </ReactMarkdown>
                  )}
                </div>

                {message.sender === "user" && (
                  <div
                    className="message-avatar user-message-avatar"
                    aria-hidden="true"
                  >
                    <User size={15} />
                  </div>
                )}
              </div>
            ))}

            {/* TYPING INDICATOR */}
            {isTyping && (
              <div
                className="chat-message bot-message"
                aria-label="Assistant is typing"
              >
                <div
                  className="message-avatar bot-message-avatar"
                  aria-hidden="true"
                >
                  <Bot size={15} />
                </div>

                <div
                  className="typing-indicator"
                  aria-hidden="true"
                >
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* ERROR STATUS */}
          {error && (
            <div className="chatbot-error" role="status">
              Connection issue. Check your backend and Ollama.
            </div>
          )}

          {/* INPUT */}
          <div className="chatbot-input-area">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about yoga..."
              aria-label="Message VITA Assistant"
              autoComplete="off"
              disabled={isTyping}
            />

            <button
              type="button"
              className="chatbot-send"
              onClick={sendMessage}
              disabled={!input.trim() || isTyping}
              aria-label="Send message"
              title="Send message"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      )}

      {/* FLOATING BUTTON */}
      <button
        type="button"
        className={`chatbot-floating-button${
          isOpen ? " chatbot-button-open" : ""
        }`}
        onClick={() => {
          if (isOpen) {
            handleClose();
          } else {
            setIsOpen(true);
          }
        }}
        aria-label={
          isOpen ? "Close VITA Assistant" : "Open VITA Assistant"
        }
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <X size={24} />
        ) : (
          <MessageCircle size={24} />
        )}
      </button>
    </>
  );
}

export default ChatBot;