"""User feedback and engagement persistence endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import InteractionCreateRequest, InteractionCreateResponse

router = APIRouter(prefix="/api", tags=["Interactions"])


@router.post("/interactions", response_model=InteractionCreateResponse)
def post_interaction(
    payload: InteractionCreateRequest,
    container: AppContainer = Depends(get_container),
) -> InteractionCreateResponse:
    article = container.repository.get_article_by_id(payload.article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article {payload.article_id} not found.")
    event = container.repository.get_event_by_id(payload.event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event '{payload.event_id}' not found.")

    now_epoch = datetime.now(timezone.utc).timestamp()
    res = container.recommendation_service.record_interaction(
        user_id=payload.user_id,
        article_id=payload.article_id,
        event_id=payload.event_id,
        interaction_type=payload.interaction_type,
        interaction_epoch=now_epoch,
    )
    return InteractionCreateResponse(**res)
