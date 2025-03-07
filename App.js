import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { FaUser, FaRobot } from "react-icons/fa"; // Import user and bot icons

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (input.trim() === "") return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const response = await axios.post("http://localhost:5000/chat", {
        query: input,
      });

      const botMessage = { text: response.data.response, sender: "bot" };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = {
        text: "An error occurred. Please try again.",
        sender: "bot",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    setInput("");
  };

  // Function to format bot responses (handling bullet points)
  const formatResponse = (text) => {
    return text.split("\n").map((line, index) => {
      if (line.startsWith("-")) {
        return <li key={index}>{line.substring(1).trim()}</li>;
      } else {
        return <p key={index}>{line}</p>;
      }
    });
  };

  return (
    <div className="chat-widget">
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message-container ${msg.sender}`}>
            {msg.sender === "user" ? <FaUser className="icon user-icon" /> : <FaRobot className="icon bot-icon" />}
            <div className={`message ${msg.sender}`}>
              {msg.sender === "bot" ? <ul>{formatResponse(msg.text)}</ul> : msg.text}
            </div>
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;
