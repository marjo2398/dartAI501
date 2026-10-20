from typing import List, Dict, Optional
from vision.dartboard import ScoreResult

class Player:
    def __init__(self, name: str, starting_score: int = 501):
        self.name = name
        self.current_score = starting_score
        self.starting_score = starting_score
        self.darts_thrown = 0
        self.legs_won = 0
        self.sets_won = 0
        self.leg_dart_scores: List[ScoreResult] = []

    @property
    def average_3dart(self) -> float:
        if self.darts_thrown == 0:
            return 0.0
        total_points = sum(d.value for d in self.leg_dart_scores)
        return round((total_points / self.darts_thrown) * 3, 2)

    def to_dict(self):
        return {
            "name": self.name,
            "current_score": self.current_score,
            "darts_thrown": self.darts_thrown,
            "legs_won": self.legs_won,
            "sets_won": self.sets_won,
            "average_3dart": self.average_3dart,
        }


class Turn:
    def __init__(self, player_name: str, start_score: int):
        self.player_name = player_name
        self.start_score = start_score
        self.darts: List[ScoreResult] = []
        self.is_bust = False
        self.is_finished = False

    def add_dart(self, score: ScoreResult, double_out: bool = True) -> int:
        """
        Adds a dart to the turn.
        Calculates remaining score, checks for bust or leg completion.
        Returns remaining score after this dart.
        """
        if self.is_finished:
            raise ValueError("Turn is already finished.")

        self.darts.append(score)
        turn_points = sum(d.value for d in self.darts)
        remaining = self.start_score - turn_points

        if remaining < 0:
            self.is_bust = True
            self.is_finished = True
            return self.start_score
        elif remaining == 1 and double_out:
            # Cannot finish on 1 with double out rule
            self.is_bust = True
            self.is_finished = True
            return self.start_score
        elif remaining == 0:
            if double_out and score.multiplier != 2:
                # Must finish on a double
                self.is_bust = True
                self.is_finished = True
                return self.start_score
            else:
                self.is_finished = True
                return 0

        if len(self.darts) == 3:
            self.is_finished = True

        return remaining


class X01Game:
    def __init__(self, player_names: List[str], target_score: int = 501, double_out: bool = True):
        if not player_names:
            raise ValueError("At least one player is required.")
        self.target_score = target_score
        self.double_out = double_out
        self.players: List[Player] = [Player(name, target_score) for name in player_names]
        self.current_player_index = 0
        self.history: List[Dict] = []
        self.winner: Optional[Player] = None
        self.current_turn = Turn(self.players[0].name, self.players[0].current_score)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def throw_dart(self, score: ScoreResult) -> Dict:
        """Processes a single dart thrown by the active player."""
        if self.winner is not None:
            return {"status": "game_over", "winner": self.winner.name}

        player = self.current_player

        # Save snapshot for undo
        snapshot = {
            "player_index": self.current_player_index,
            "player_score": player.current_score,
            "player_darts_count": player.darts_thrown,
            "turn_darts": list(self.current_turn.darts),
            "turn_is_bust": self.current_turn.is_bust,
            "turn_is_finished": self.current_turn.is_finished,
            "winner": self.winner.name if self.winner else None
        }
        self.history.append(snapshot)

        new_remaining = self.current_turn.add_dart(score, double_out=self.double_out)
        player.darts_thrown += 1
        player.leg_dart_scores.append(score)

        if not self.current_turn.is_bust:
            player.current_score = new_remaining

        if player.current_score == 0:
            player.legs_won += 1
            self.winner = player

        is_turn_end = self.current_turn.is_finished
        last_dart = score.to_dict()

        if is_turn_end and self.winner is None:
            self.next_turn()

        return {
            "status": "game_over" if self.winner else ("turn_ended" if is_turn_end else "in_progress"),
            "current_player": player.name,
            "last_dart": last_dart,
            "turn_darts": [d.to_dict() for d in self.current_turn.darts] if not is_turn_end else [],
            "turn_bust": self.current_turn.is_bust,
            "remaining_score": player.current_score,
            "winner": self.winner.name if self.winner else None
        }

    def next_turn(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.current_turn = Turn(self.current_player.name, self.current_player.current_score)

    def undo_last_dart(self) -> bool:
        """Undoes the last thrown dart."""
        if not self.history:
            return False

        snapshot = self.history.pop()
        player = self.players[snapshot["player_index"]]

        # Revert player state
        player.current_score = snapshot["player_score"]
        if player.darts_thrown > 0:
            player.darts_thrown -= 1
        if player.leg_dart_scores:
            player.leg_dart_scores.pop()

        self.current_player_index = snapshot["player_index"]
        self.winner = None

        self.current_turn = Turn(player.name, snapshot["player_score"])
        self.current_turn.darts = snapshot["turn_darts"]
        self.current_turn.is_bust = snapshot["turn_is_bust"]
        self.current_turn.is_finished = snapshot["turn_is_finished"]

        return True

    def get_state(self) -> Dict:
        return {
            "target_score": self.target_score,
            "double_out": self.double_out,
            "players": [p.to_dict() for p in self.players],
            "current_player_index": self.current_player_index,
            "current_player_name": self.current_player.name,
            "current_turn_darts": [d.to_dict() for d in self.current_turn.darts],
            "winner": self.winner.name if self.winner else None,
            "checkout_recommendation": self.get_checkout_recommendation(self.current_player.current_score)
        }

    @staticmethod
    def get_checkout_recommendation(score: int) -> Optional[str]:
        """Provides standard 2-dart / 3-dart checkout suggestions."""
        CHECKOUTS = {
            170: "T20 T20 Bull", 167: "T20 T19 Bull", 164: "T20 T18 Bull",
            161: "T20 T17 Bull", 160: "T20 T20 D20", 150: "T20 T18 D18",
            100: "T20 D20", 80: "T20 D10", 60: "20 D20", 50: "Bull",
            40: "D20", 36: "D18", 32: "D16", 24: "D12", 16: "D8", 8: "D4", 4: "D2", 2: "D1"
        }
        return CHECKOUTS.get(score, None)
