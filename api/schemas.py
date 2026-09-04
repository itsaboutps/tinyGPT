from pydantic import (
    BaseModel,
    Field,
)


class ChatRequest(
    BaseModel
):

    message: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(
    BaseModel
):

    response: str


class HealthResponse(
    BaseModel
):

    status: str

    model_loaded: bool

    device: str

    checkpoint: str


class CreateConversationResponse(
    BaseModel
):

    conversation_id: str


class SendMessageRequest(
    BaseModel
):

    message: str = Field(
        min_length=1,
        max_length=2000,
    )


class SendMessageResponse(
    BaseModel
):

    conversation_id: str

    response: str


class MessageResponse(
    BaseModel
):

    role: str

    content: str


class ConversationResponse(
    BaseModel
):

    conversation_id: str

    messages: list[
        MessageResponse
    ]