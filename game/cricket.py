from typing import List, Dict, Optional
from vision.dartboard import ScoreResult

SECTORS_CRICKET = [20, 19, 18, 17, 16, 15, 25]

class CricketGame:
    def __init__(self, player_names: List[str]):
        if not player_names:
            raise ValueError("At least 1 player required.")
        self.player_names = player_names
        self.scores = {p: 0 for p in player_names}
        # Marks for sectors 15-20 and 25 (Bull)
        self.marks = {p: {s: 0 for s in SECTORS_CRICKET} for p in player_names}
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
        sector = score.sector
        multiplier = score.multiplier

        if sector in SECTORS_CRICKET and multiplier > 0:
            current_marks = self.marks[player][sector]
            new_marks = current_marks + multiplier

            if current_marks < 3:
                marks_needed_to_close = 3 - current_marks
                actual_marks_applied = min(multiplier, marks_needed_to_close)
                overflow = multiplier - actual_marks_applied

                self.marks[player][sector] += actual_marks_applied

                if overflow > 0 and not self.is_closed_by_all(sector):
                    val = 25 if sector == 25 else sector
                    self.scores[player] += overflow * val
            else:
                if not self.is_closed_by_all(sector):
                    val = 25 if sector == 25 else sector
                    self.scores[player] += multiplier * val

        self.darts_in_turn += 1

        # Check win condition: all sectors closed and highest or equal score
        if self.has_closed_all(player) and self.scores[player] >= max(self.scores.values()):
            self.winner = player

        is_turn_end = False
        if self.darts_in_turn == 3:
            is_turn_end = True
            self.next_turn()

        return {
            "status": "game_over" if self.winner else ("turn_ended" if is_turn_end else "in_progress"),
            "current_player": player,
            "last_dart": score.to_dict(),
            "winner": self.winner
        }

    def is_closed_by_all(self, sector: int) -> bool:
        return all(self.marks[p][sector] >= 3 for p in self.player_names)

    def has_closed_all(self, player: str) -> bool:
        return all(self.marks[player][s] >= 3 for s in SECTORS_CRICKET)

    def next_turn(self):
        self.darts_in_turn = 0
        self.current_player_index = (self.current_player_index + 1) % len(self.player_names)

    def get_state(self) -> Dict:
        return {
            "game_type": "Cricket",
            "players": self.player_names,
            "current_player": self.current_player,
            "scores": self.scores,
            "marks": self.marks,
            "winner": self.winner
        }
