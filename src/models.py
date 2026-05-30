"""Pydantic data model for a single flower manual."""
from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


def _coerce_int(v):
    """Accept ints, numeric strings, or fuzzy strings like 'many (80-120)'."""
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        nums = [int(n) for n in re.findall(r"\d+", v)]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) // 2  # midpoint of a range
        if nums:
            return nums[0]
    return v  # let pydantic raise its normal error


class PetalLayer(BaseModel):
    """One ring/whorl of petals on the flower."""
    name: str = Field(..., description='e.g. "outer", "middle", "inner"')
    count: int = Field(..., description="Number of petals in this layer")

    @field_validator("count", mode="before")
    @classmethod
    def _parse_count(cls, v):
        return _coerce_int(v)
    relative_size: Optional[str] = Field(
        None, description='e.g. "small", "medium", "big" or relative to other layers'
    )
    shape: Optional[str] = Field(
        None,
        description='Petal shape, e.g. "rounded", "pointed", "ruffled", "fringed"',
    )
    color: Optional[str] = Field(
        None, description='Hex or descriptive color, e.g. "#e8c85a / pale yellow"'
    )
    notes: Optional[str] = None


class Anatomy(BaseModel):
    """Structured anatomy of the photographed flower.

    Filled by the AI from the photo, then used to derive bead counts
    and component structure. Editable in the UI.
    """
    flower_family: Optional[str] = Field(
        None, description='e.g. "Asteraceae (daisy family)"'
    )
    overall_shape: Optional[str] = Field(
        None, description='e.g. "flat composite", "cup", "bell", "spike", "globe"'
    )
    petal_layers: List[PetalLayer] = Field(
        default_factory=list,
        description="Ordered outer-to-inner. Sum of counts = total petals.",
    )
    total_petals: Optional[int] = Field(
        None, description="Sum of petal counts across layers"
    )
    sepal_count: Optional[int] = None
    has_center_disc: bool = Field(
        False, description="True for daisies/coneflowers/sunflowers"
    )
    center_description: Optional[str] = Field(
        None, description='e.g. "yellow disc with stamens", "dark cone with spikes"'
    )
    has_stamens: bool = False
    leaf_arrangement: Optional[str] = Field(
        None, description='e.g. "alternate, lanceolate", "opposite, ovate"'
    )
    leaf_count_estimate: Optional[int] = None
    stem_height_cm_estimate: Optional[float] = None
    distinguishing_features: List[str] = Field(
        default_factory=list,
        description="Anything beader-relevant: ruffled edges, bicolor petals, etc.",
    )

    @field_validator(
        "total_petals", "sepal_count", "leaf_count_estimate", mode="before"
    )
    @classmethod
    def _parse_optional_int(cls, v):
        if v is None or v == "":
            return None
        return _coerce_int(v)


class Material(BaseModel):
    name: str
    quantity: Optional[str] = None  # e.g. "5g", "50cm", "1 spool"


class ComponentImage(BaseModel):
    path: str          # local file path or URL
    caption: Optional[str] = None


class Component(BaseModel):
    """A part of the flower: petals, center, leaves, etc."""
    heading: str = Field(
        ...,
        description=(
            'Plain-English title naming THIS specific flower\'s part, ending '
            'with the count, e.g. "Outer spiky leaves (24x):" for an artichoke '
            'or "White ray petals (21x):" for a daisy. Avoid jargon like '
            '"bracts", "calyx", "ray florets".'
        ),
    )
    plain_description: Optional[str] = Field(
        None,
        description=(
            'One short sentence telling the maker what this part IS in plain '
            'language, e.g. "These are the long pointed leaves that wrap '
            'around the artichoke head."'
        ),
    )
    count_label: Optional[str] = Field(
        None, description='e.g. "Small (4x), Medium (4x), Big (5x)"'
    )
    paragraphs: List[str] = Field(default_factory=list)
    tip: Optional[str] = None
    note: Optional[str] = None
    images: List[ComponentImage] = Field(default_factory=list)


class AssemblySection(BaseModel):
    heading: str = "Assembling the Stem:"
    paragraphs: List[str] = Field(default_factory=list)
    images: List[ComponentImage] = Field(default_factory=list)


class Flower(BaseModel):
    """One complete flower manual."""

    # Cover / intro
    name: str = Field(..., description='e.g. "Daisy"')
    title: str = Field(..., description='e.g. "French Beaded Daisy"')
    author: str = "Henri Purnell"
    intro: str = ""
    guided_video_url: Optional[str] = None
    hero_image: Optional[str] = None

    # Color palette suggested from the photo (hex strings, used for accents)
    palette: List[str] = Field(default_factory=list)

    # Structured anatomy — drives realistic bead counts
    anatomy: Optional[Anatomy] = None

    # 2-3 sentence visual brief used to ground EVERY illustration prompt.
    # Describes the overall silhouette, petal-stacking pattern, density,
    # proportions, and where colour transitions happen, so gpt-image-1 doesn't
    # default to a generic "flower" shape.
    visual_summary: Optional[str] = Field(
        None,
        description=(
            "2-3 sentences describing the visual character of THIS specific "
            "flower from the photo: silhouette, how petals stack/overlap, "
            "density, proportions, colour transitions. Used verbatim in "
            "image-generation prompts. Example for a shampoo ginger: 'A "
            "dense torpedo-shaped flower head with overlapping waxy bracts "
            "stacked like roof shingles, wider at the base than the tip. "
            "Magenta-pink at the base fading to pale yellow-green at the "
            "growing tip. Compact and chunky, NOT a fanned-out daisy shape.'"
        ),
    )

    # Single labeled hand-drawn diagram showing each component the maker will
    # build (with arrows naming them in plain English).
    anatomy_diagram: Optional[ComponentImage] = None

    # Materials list (rendered as two-column block on the cover page)
    materials: List[Material] = Field(default_factory=list)

    # Difficulty / counts shown on the cover (e.g. "Beginner", "Makes 1 flower")
    difficulty: Optional[str] = None
    yields: Optional[str] = None  # e.g. "Makes 1 flower"

    # Step-by-step components
    components: List[Component] = Field(default_factory=list)

    # Final assembly
    assembly: Optional[AssemblySection] = None

    # Inspo gallery at the back
    inspo_images: List[ComponentImage] = Field(default_factory=list)

    # Footer / copyright (defaults sensible)
    copyright_year: int = 2026
    copyright_holder: str = "Henri Purnell"
    copyright_text: Optional[str] = None
