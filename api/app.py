import asyncio
import json

from contextlib import (
    asynccontextmanager,
)

from pathlib import (
    Path,
)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from tinygpt.api.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)

from tinygpt.api.model_service import (
    ChatModelService,
)

from tinygpt.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    CreateConversationResponse,
    HealthResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)

from tinygpt.sft.formatting import (
    ChatMessage,
)


#
# ============================================================
# PATHS
# ============================================================
#

CHECKPOINT_PATH = (
    "checkpoints/"
    "tinychat_sft_v1/"
    "best_chat.pt"
)


TOKENIZER_PATH = (
    "data/tokenizer/"
    "tokenizer.json"
)


#
# app.py is located at:
#
# tinygpt/
# └── tinygpt/
#     └── api/
#         └── app.py
#
# parents[2] therefore gives us
# the repository root.
#

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


FRONTEND_DIR = (
    PROJECT_ROOT
    /
    "frontend"
)


#
# ============================================================
# APPLICATION LIFESPAN
# ============================================================
#

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "Loading TinyChatGPT..."
    )


    #
    # Create the model service.
    #

    chat_service = (
        ChatModelService(
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            tokenizer_path=(
                TOKENIZER_PATH
            ),
        )
    )


    #
    # Load tokenizer + model once.
    #
    # They remain in memory for the
    # lifetime of the FastAPI process.
    #

    chat_service.load()


    #
    # Create in-memory conversation
    # storage.
    #

    conversation_service = (
        ConversationService()
    )


    #
    # Store application-level services
    # on FastAPI state so all routes
    # can reuse the same instances.
    #

    app.state.chat_service = (
        chat_service
    )


    app.state.conversation_service = (
        conversation_service
    )


    print(
        "TinyChatGPT loaded."
    )


    #
    # Everything before yield runs
    # during application startup.
    #
    # The server accepts requests while
    # execution is paused here.
    #

    yield


    #
    # Everything after yield runs
    # during application shutdown.
    #

    print(
        "TinyChatGPT shutting down."
    )


#
# ============================================================
# FASTAPI APPLICATION
# ============================================================
#

app = FastAPI(
    title="TinyChatGPT API",
    version="1.0.0",
    lifespan=lifespan,
)


#
# ============================================================
# STATIC FRONTEND
# ============================================================
#

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            FRONTEND_DIR
        )
    ),
    name="static",
)


#
# ============================================================
# FRONTEND ROUTE
# ============================================================
#

@app.get(
    "/",
    include_in_schema=False,
)
async def frontend():

    return FileResponse(
        FRONTEND_DIR
        /
        "index.html"
    )


#
# ============================================================
# HEALTH
# ============================================================
#

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health(
    request: Request,
):

    service = (
        request.app.state
        .chat_service
    )


    return HealthResponse(
        status="healthy",
        model_loaded=(
            service.is_loaded
        ),
        device=str(
            service.device
        ),
        checkpoint=(
            service.checkpoint_path
        ),
    )


#
# ============================================================
# STATELESS CHAT
# ============================================================
#
# This endpoint does NOT retain
# conversation history.
#
# It is useful for testing/debugging.
#

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    request: Request,
):

    service = (
        request.app.state
        .chat_service
    )


    try:

        response = (
            await asyncio.to_thread(
                service.chat,
                payload.message,
            )
        )


        return ChatResponse(
            response=response
        )


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "TinyChatGPT inference failed"
            ),
        ) from exc


#
# ============================================================
# CREATE CONVERSATION
# ============================================================
#

@app.post(
    "/conversations",
    response_model=(
        CreateConversationResponse
    ),
)
async def create_conversation(
    request: Request,
):

    service = (
        request.app.state
        .conversation_service
    )


    conversation_id = (
        service.create_conversation()
    )


    return (
        CreateConversationResponse(
            conversation_id=(
                conversation_id
            )
        )
    )


#
# ============================================================
# GET CONVERSATION
# ============================================================
#

@app.get(
    "/conversations/{conversation_id}",
    response_model=(
        ConversationResponse
    ),
)
async def get_conversation(
    conversation_id: str,
    request: Request,
):

    service = (
        request.app.state
        .conversation_service
    )


    try:

        messages = (
            service.get_messages(
                conversation_id
            )
        )


    except ConversationNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        ) from exc


    return ConversationResponse(
        conversation_id=(
            conversation_id
        ),
        messages=[
            MessageResponse(
                role=message.role,
                content=(
                    message.content
                ),
            )
            for message
            in messages
        ],
    )


#
# ============================================================
# NORMAL MULTI-TURN CHAT
# ============================================================
#
# This endpoint waits for the complete
# model response before returning JSON.
#

@app.post(
    "/conversations/"
    "{conversation_id}/messages",
    response_model=(
        SendMessageResponse
    ),
)
async def send_message(
    conversation_id: str,
    payload: SendMessageRequest,
    request: Request,
):

    chat_service = (
        request.app.state
        .chat_service
    )


    conversation_service = (
        request.app.state
        .conversation_service
    )


    try:

        #
        # Retrieve previous conversation
        # history.
        #

        previous_messages = (
            conversation_service
            .get_messages(
                conversation_id
            )
        )


        #
        # The current user message has not
        # yet been permanently committed.
        #
        # Add it temporarily so the model
        # can see it.
        #

        current_messages = (
            previous_messages
            +
            [
                ChatMessage(
                    role="user",
                    content=(
                        payload.message
                    ),
                )
            ]
        )


        #
        # Model inference is synchronous
        # CPU work.
        #
        # Run it outside FastAPI's event
        # loop.
        #

        response = (
            await asyncio.to_thread(
                chat_service
                .chat_with_history,
                current_messages,
            )
        )


        #
        # Only store the user + assistant
        # exchange after generation succeeds.
        #

        conversation_service.append_exchange(
            conversation_id=(
                conversation_id
            ),
            user_message=(
                payload.message
            ),
            assistant_message=(
                response
            ),
        )


        return SendMessageResponse(
            conversation_id=(
                conversation_id
            ),
            response=response,
        )


    except ConversationNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        ) from exc


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "TinyChatGPT inference failed"
            ),
        ) from exc


#
# ============================================================
# STREAMING MULTI-TURN CHAT
# ============================================================
#
# Instead of waiting for the entire
# assistant response, this endpoint sends
# generated text chunks progressively.
#
# Response format:
#
# event: token
# data: {"type":"token","text":"Hello"}
#
# event: done
# data: {"type":"done", ...}
#

@app.post(
    "/conversations/"
    "{conversation_id}/messages/stream"
)
async def stream_message(
    conversation_id: str,
    payload: SendMessageRequest,
    request: Request,
):

    chat_service = (
        request.app.state
        .chat_service
    )


    conversation_service = (
        request.app.state
        .conversation_service
    )


    message = (
        payload.message.strip()
    )


    if not message:

        raise HTTPException(
            status_code=400,
            detail=(
                "message cannot be empty"
            ),
        )


    #
    # IMPORTANT:
    #
    # Validate conversation existence
    # BEFORE StreamingResponse starts.
    #
    # Once streaming begins, HTTP headers
    # may already have been sent and we
    # cannot later change the HTTP status
    # to 404.
    #

    try:

        previous_messages = (
            conversation_service
            .get_messages(
                conversation_id
            )
        )


    except ConversationNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        ) from exc


    #
    # Build temporary history:
    #
    # previous messages
    # +
    # current user message
    #

    current_messages = (
        previous_messages
        +
        [
            ChatMessage(
                role="user",
                content=message,
            )
        ]
    )


    #
    # This generator produces SSE-style
    # HTTP events.
    #

    def event_stream():

        response_chunks = []


        try:

            #
            # The model yields decoded text
            # progressively.
            #

            for chunk in (
                chat_service
                .stream_chat_with_history(
                    current_messages
                )
            ):

                response_chunks.append(
                    chunk
                )


                event = {
                    "type": "token",
                    "text": chunk,
                }


                yield (
                    "event: token\n"
                    +
                    "data: "
                    +
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    +
                    "\n\n"
                )


            #
            # Reconstruct the complete
            # assistant response.
            #

            full_response = (
                "".join(
                    response_chunks
                )
                .strip()
            )


            if not full_response:

                full_response = (
                    "[No response generated]"
                )


            #
            # Commit the complete exchange
            # only after successful model
            # generation.
            #

            conversation_service.append_exchange(
                conversation_id=(
                    conversation_id
                ),
                user_message=(
                    message
                ),
                assistant_message=(
                    full_response
                ),
            )


            #
            # Tell the frontend that
            # generation is complete.
            #

            done_event = {
                "type": "done",
                "conversation_id": (
                    conversation_id
                ),
            }


            yield (
                "event: done\n"
                +
                "data: "
                +
                json.dumps(
                    done_event,
                    ensure_ascii=False,
                )
                +
                "\n\n"
            )


        except Exception as exc:

            #
            # Once streaming has already
            # started, changing HTTP status
            # to 500 may no longer be
            # possible.
            #
            # Therefore send an application
            # level error event.
            #

            print(
                "Streaming error:",
                repr(exc),
            )


            error_event = {
                "type": "error",
                "message": (
                    "TinyChatGPT "
                    "streaming failed"
                ),
            }


            yield (
                "event: error\n"
                +
                "data: "
                +
                json.dumps(
                    error_event,
                    ensure_ascii=False,
                )
                +
                "\n\n"
            )


    #
    # Keep the HTTP connection open and
    # progressively send each yielded
    # SSE event.
    #

    return StreamingResponse(
        event_stream(),
        media_type=(
            "text/event-stream"
        ),
        headers={
            "Cache-Control": (
                "no-cache"
            ),
            "X-Accel-Buffering": (
                "no"
            ),
        },
    )