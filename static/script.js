document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const chatBox = document.getElementById("chat-box");
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.innerHTML = "<span></span><span></span><span></span>";

    const appendMessage = (message, sender) => {
        const messageElement = document.createElement("div");
        messageElement.className = `chat-message ${sender}-message`;
        messageElement.innerText = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const showTypingIndicator = () => {
        chatBox.appendChild(typingIndicator);
        typingIndicator.style.display = "flex";
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const hideTypingIndicator = () => {
        if (chatBox.contains(typingIndicator)) {
            chatBox.removeChild(typingIndicator);
        }
    };

    const sendMessage = async () => {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;

        appendMessage(userMessage, "user");
        userInput.value = "";
        showTypingIndicator();

        try {
            const response = await fetch("/get", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ msg: userMessage }),
            });
            const data = await response.json();
            hideTypingIndicator();
            appendMessage(data.answer, "bot");
        } catch (error) {
            hideTypingIndicator();
            appendMessage("Sorry, something went wrong.", "bot");
        }
    };

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
