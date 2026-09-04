const chatContainer =
    document.getElementById(
        "chat-container"
    );

const welcomePanel =
    document.getElementById(
        "welcome-panel"
    );

const messageForm =
    document.getElementById(
        "message-form"
    );

const messageInput =
    document.getElementById(
        "message-input"
    );

const sendButton =
    document.getElementById(
        "send-button"
    );

const newChatButton =
    document.getElementById(
        "new-chat-button"
    );

const statusText =
    document.getElementById(
        "status-text"
    );

const conversationInfo =
    document.getElementById(
        "conversation-info"
    );


let conversationId = null;

let isGenerating = false;



async function createConversation() {

    statusText.textContent =
        "Creating conversation...";


    const response = await fetch(
        "/conversations",
        {
            method: "POST",
        }
    );


    if (!response.ok) {

        throw new Error(
            "Failed to create conversation"
        );
    }


    const data =
        await response.json();


    conversationId =
        data.conversation_id;


    conversationInfo.textContent =
        `Conversation: ${conversationId}`;


    statusText.textContent =
        "Ready";
}



function clearMessages() {

    const existingMessages =
        chatContainer.querySelectorAll(
            ".message-row"
        );


    existingMessages.forEach(
        element => element.remove()
    );


    welcomePanel.style.display =
        "block";
}



function hideWelcomePanel() {

    welcomePanel.style.display =
        "none";
}



function appendMessage(
    role,
    text
) {

    hideWelcomePanel();


    const row =
        document.createElement(
            "section"
        );


    row.classList.add(
        "message-row",
        role
    );


    const roleElement =
        document.createElement(
            "div"
        );


    roleElement.className =
        "message-role";


    roleElement.textContent =
        role === "user"
            ? "You"
            : "TinyChatGPT";


    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "message-bubble";


    bubble.textContent =
        text;


    row.appendChild(
        roleElement
    );


    row.appendChild(
        bubble
    );


    chatContainer.appendChild(
        row
    );


    scrollToBottom();


    return bubble;
}



function scrollToBottom() {

    chatContainer.scrollTo({
        top:
            chatContainer.scrollHeight,
        behavior: "smooth",
    });
}



function setGenerating(
    generating
) {

    isGenerating =
        generating;


    sendButton.disabled =
        generating;


    newChatButton.disabled =
        generating;


    messageInput.disabled =
        generating;


    statusText.textContent =
        generating
            ? "TinyChatGPT is generating..."
            : "Ready";
}



function parseSSEBlock(
    block
) {

    const lines =
        block.split("\n");


    let eventName = null;

    let dataText = "";


    for (const line of lines) {

        if (
            line.startsWith(
                "event:"
            )
        ) {

            eventName =
                line
                    .slice(
                        "event:".length
                    )
                    .trim();
        }


        if (
            line.startsWith(
                "data:"
            )
        ) {

            const value =
                line
                    .slice(
                        "data:".length
                    )
                    .trim();


            dataText += value;
        }
    }


    if (!dataText) {

        return null;
    }


    let data;


    try {

        data =
            JSON.parse(
                dataText
            );

    } catch {

        return null;
    }


    return {
        event: eventName,
        data: data,
    };
}



async function streamMessage(
    message
) {

    if (!conversationId) {

        await createConversation();
    }


    const response = await fetch(
        `/conversations/${conversationId}/messages/stream`,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",
            },

            body: JSON.stringify({
                message: message,
            }),
        }
    );


    if (!response.ok) {

        let detail =
            "Request failed";


        try {

            const errorBody =
                await response.json();


            if (errorBody.detail) {

                detail =
                    errorBody.detail;
            }

        } catch {
        }


        throw new Error(
            detail
        );
    }


    if (!response.body) {

        throw new Error(
            "Streaming response body "
            +
            "is unavailable"
        );
    }


    const assistantBubble =
        appendMessage(
            "assistant",
            ""
        );


    assistantBubble.classList.add(
        "assistant-cursor"
    );


    const reader =
        response.body.getReader();


    const decoder =
        new TextDecoder(
            "utf-8"
        );


    let buffer = "";

    let fullAssistantText = "";


    try {

        while (true) {

            const {
                value,
                done,
            } = await reader.read();


            if (done) {

                break;
            }


            buffer +=
                decoder.decode(
                    value,
                    {
                        stream: true,
                    }
                );


            buffer =
                buffer.replace(
                    /\r\n/g,
                    "\n"
                );


            let separatorIndex;


            while (
                (
                    separatorIndex =
                        buffer.indexOf(
                            "\n\n"
                        )
                )
                !==
                -1
            ) {

                const block =
                    buffer.slice(
                        0,
                        separatorIndex
                    );


                buffer =
                    buffer.slice(
                        separatorIndex + 2
                    );


                const event =
                    parseSSEBlock(
                        block
                    );


                if (!event) {

                    continue;
                }


                if (
                    event.event ===
                    "token"
                ) {

                    const tokenText =
                        event.data.text
                        ?? "";


                    fullAssistantText +=
                        tokenText;


                    assistantBubble.textContent =
                        fullAssistantText;


                    scrollToBottom();
                }


                if (
                    event.event ===
                    "done"
                ) {

                    statusText.textContent =
                        "Ready";
                }


                if (
                    event.event ===
                    "error"
                ) {

                    throw new Error(
                        event.data.message
                        ||
                        "Streaming failed"
                    );
                }
            }
        }


        buffer +=
            decoder.decode();


        if (
            !fullAssistantText.trim()
        ) {

            assistantBubble.textContent =
                "[No response generated]";
        }

    } finally {

        assistantBubble
            .classList
            .remove(
                "assistant-cursor"
            );
    }
}



async function sendCurrentMessage() {

    if (isGenerating) {

        return;
    }


    const message =
        messageInput
            .value
            .trim();


    if (!message) {

        return;
    }


    appendMessage(
        "user",
        message
    );


    messageInput.value =
        "";


    setGenerating(
        true
    );


    try {

        await streamMessage(
            message
        );

    } catch (error) {

        console.error(
            error
        );


        const errorBubble =
            appendMessage(
                "assistant",
                `Error: ${error.message}`
            );


        errorBubble.classList.add(
            "error-message"
        );


        statusText.textContent =
            "Error";

    } finally {

        setGenerating(
            false
        );


        messageInput.focus();
    }
}



messageForm.addEventListener(
    "submit",
    async event => {

        event.preventDefault();


        await sendCurrentMessage();
    }
);



messageInput.addEventListener(
    "keydown",
    async event => {

        if (
            event.key === "Enter"
            &&
            !event.shiftKey
        ) {

            event.preventDefault();


            await sendCurrentMessage();
        }
    }
);



messageInput.addEventListener(
    "input",
    () => {

        messageInput.style.height =
            "auto";


        messageInput.style.height =
            `${Math.min(
                messageInput.scrollHeight,
                160
            )}px`;
    }
);



newChatButton.addEventListener(
    "click",
    async () => {

        if (isGenerating) {

            return;
        }


        try {

            clearMessages();


            conversationId =
                null;


            conversationInfo.textContent =
                "";


            await createConversation();


            messageInput.focus();

        } catch (error) {

            console.error(
                error
            );


            statusText.textContent =
                "Failed to create chat";
        }
    }
);



async function initialize() {

    try {

        const healthResponse =
            await fetch(
                "/health"
            );


        if (!healthResponse.ok) {

            throw new Error(
                "Backend health check failed"
            );
        }


        const health =
            await healthResponse.json();


        if (
            !health.model_loaded
        ) {

            throw new Error(
                "Model is not loaded"
            );
        }


        await createConversation();


        messageInput.focus();

    } catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "Backend unavailable";


        sendButton.disabled =
            true;
    }
}



initialize();