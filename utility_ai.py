"""A small, extensible Utility AI model for game NPC decision making.

Utility AI scores each possible action from the current world state and picks the
highest utility option. This example models a survival NPC that balances hunger,
thirst, danger, tiredness, and social needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Callable, Iterable


@dataclass(frozen=True)
class WorldState:
    """Normalized facts about the NPC and its environment.

    All values are expected to be in the range 0.0 to 1.0, where 0.0 is low and
    1.0 is high. For example, ``hunger=1.0`` means the NPC is starving.
    """

    hunger: float
    thirst: float
    danger: float
    tiredness: float
    loneliness: float
    food_nearby: float
    water_nearby: float
    ally_nearby: float
    shelter_nearby: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


@dataclass(frozen=True)
class Consideration:
    """A single scoring rule used by an action.

    The ``curve`` function converts a world-state value into utility. Curves make
    the agent feel smarter than plain if/else checks because each need can become
    urgent gradually or suddenly depending on the chosen curve.
    """

    name: str
    read_value: Callable[[WorldState], float]
    curve: Callable[[float], float]
    weight: float = 1.0

    def score(self, state: WorldState) -> float:
        raw_value = clamp01(self.read_value(state))
        curved_value = clamp01(self.curve(raw_value))
        return clamp01(curved_value * self.weight)


@dataclass(frozen=True)
class Action:
    """An NPC action with several considerations that produce one utility score."""

    name: str
    considerations: tuple[Consideration, ...]
    description: str
    inertia_bonus: float = 0.05

    def utility(self, state: WorldState, previous_action: str | None = None) -> float:
        if not self.considerations:
            return 0.0

        # Multiplication means one critical blocker can strongly lower the action.
        # The compensation factor prevents actions with many considerations from
        # becoming unfairly tiny compared with simpler actions.
        scores = [consideration.score(state) for consideration in self.considerations]
        compensated_score = prod(scores) ** (1 / len(scores))

        if previous_action == self.name:
            compensated_score += self.inertia_bonus

        return clamp01(compensated_score)


@dataclass
class UtilityAI:
    """Selects the best action by utility score and explains the decision."""

    actions: tuple[Action, ...]
    previous_action: str | None = None
    last_scores: dict[str, float] = field(default_factory=dict)

    def choose_action(self, state: WorldState) -> Action:
        if not self.actions:
            raise ValueError("UtilityAI needs at least one action")

        self.last_scores = {
            action.name: action.utility(state, self.previous_action)
            for action in self.actions
        }
        best_action = max(self.actions, key=lambda action: self.last_scores[action.name])
        self.previous_action = best_action.name
        return best_action

    def explain(self) -> str:
        if not self.last_scores:
            return "No decision has been made yet."

        ranked_scores = sorted(self.last_scores.items(), key=lambda item: item[1], reverse=True)
        return "\n".join(f"{name:14s} utility={score:.3f}" for name, score in ranked_scores)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def linear(value: float) -> float:
    return value


def inverse(value: float) -> float:
    return 1.0 - value


def quadratic(value: float) -> float:
    return value * value


def steep(value: float) -> float:
    """A curve that ignores mild needs but reacts strongly to high values."""

    return value**4


def build_survival_ai() -> UtilityAI:
    """Build an example Utility AI brain for a survival-game NPC."""

    return UtilityAI(
        actions=(
            Action(
                name="Run to shelter",
                description="Escape danger when threats are high and shelter is reachable.",
                considerations=(
                    Consideration("danger urgency", lambda state: state.danger, steep, 1.6),
                    Consideration("shelter access", lambda state: state.shelter_nearby, linear),
                    Consideration("not exhausted", lambda state: state.tiredness, inverse, 0.9),
                ),
                inertia_bonus=0.10,
            ),
            Action(
                name="Drink water",
                description="Handle thirst, especially when water is nearby.",
                considerations=(
                    Consideration("thirst urgency", lambda state: state.thirst, quadratic, 1.4),
                    Consideration("water access", lambda state: state.water_nearby, linear),
                    Consideration("safe enough", lambda state: state.danger, inverse, 0.8),
                ),
            ),
            Action(
                name="Eat food",
                description="Handle hunger, especially when food is nearby.",
                considerations=(
                    Consideration("hunger urgency", lambda state: state.hunger, quadratic, 1.3),
                    Consideration("food access", lambda state: state.food_nearby, linear),
                    Consideration("safe enough", lambda state: state.danger, inverse, 0.8),
                ),
            ),
            Action(
                name="Rest",
                description="Recover energy when tired and the area is safe.",
                considerations=(
                    Consideration("tiredness", lambda state: state.tiredness, quadratic, 1.2),
                    Consideration("safety", lambda state: state.danger, inverse),
                ),
            ),
            Action(
                name="Find ally",
                description="Seek social support when lonely and allies are nearby.",
                considerations=(
                    Consideration("loneliness", lambda state: state.loneliness, linear),
                    Consideration("ally access", lambda state: state.ally_nearby, linear),
                    Consideration("safe enough", lambda state: state.danger, inverse, 0.7),
                ),
            ),
            Action(
                name="Explore",
                description="Default behavior when no urgent need dominates.",
                considerations=(
                    Consideration("low hunger", lambda state: state.hunger, inverse),
                    Consideration("low thirst", lambda state: state.thirst, inverse),
                    Consideration("low danger", lambda state: state.danger, inverse),
                    Consideration("enough energy", lambda state: state.tiredness, inverse),
                ),
            ),
        )
    )


def run_demo(states: Iterable[WorldState]) -> None:
    ai = build_survival_ai()

    for turn, state in enumerate(states, start=1):
        action = ai.choose_action(state)
        print(f"\nTurn {turn}: {action.name}")
        print(f"Reason: {action.description}")
        print(ai.explain())


if __name__ == "__main__":
    demo_states = (
        WorldState(0.2, 0.3, 0.1, 0.2, 0.4, 0.7, 0.7, 0.5, 0.4),
        WorldState(0.4, 0.9, 0.2, 0.3, 0.2, 0.5, 0.9, 0.4, 0.3),
        WorldState(0.5, 0.4, 0.95, 0.2, 0.3, 0.5, 0.4, 0.2, 0.9),
        WorldState(0.3, 0.2, 0.1, 0.95, 0.6, 0.4, 0.4, 0.8, 0.6),
    )
    run_demo(demo_states)
