from typing import List, Dict, Optional
from vision.dartboard import ScoreResult

class AroundTheClockGame:
    def __init__(self, player_names: List[str]):
        if not player_names:
            raise ValueError("At least 1 player required.")
        self.player_names = player_names
        # Target sector for each player starts at 1 and goes up to 20, then 25 (Bull)
        self.targets = {p: 1 for p in player_names}
        self.current_player_index = 0
        self.darts_in_turn = 0
        self.winner: Optional[str] = None

    @property
    def current_player(self) -> str:
        return self.player_names[self.current_player_index]

    def throw_dart(self, score: ScoreResult) -> Dict:
        if self.winner is not None:
            return {"status": "game_over", "winner": self.winner}

        player = self.current_player
        target = self.targets[player]

        if score.sector == target and score.multiplier > 0:
            if target < 20:
                self.targets[player] += 1
            elif target == 20:
                self.targets[player] = 25  # Final Bullseye target
            elif target == 25:
                self.winner = player

        self.darts_in_turn += 1
        is_turn_end = False
        if self.darts_in_turn == 3:
            is_turn_end = True
            self.next_turn()

        return {
            "status": "game_over" if self.winner else ("turn_ended" if is_turn_end else "in_progress"),
            "current_player": player,
            "target_sector": self.targets[player],
            "last_dart": score.to_dict(),
            "winner": self.winner
        }

    def next_turn(self):
        self.darts_in_turn = 0
        self.current_player_index = (self.current_player_index + 1) % len(self.player_names)

    def get_state(self) -> Dict:
        return {
            "game_type": "AroundTheClock",
            "players": self.player_names,
            "current_player": self.current_player,
            "targets": self.targets,
            "winner": self.winner
        }
