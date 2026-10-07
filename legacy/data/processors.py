from threading import Thread
from core import constants
from pathlib import Path
import random
from typing import Union, Any, List
from core.interfaces import IProcess, IProcessor
from data.processes import (
    AlefFariqaDropper,
    AlefMaqsuraToYa,
    HamzaAlefDropper,
    HamzaSeatDropper,
    RandomCharRemover,
    RandomCharsInjector,
    RandomCharsSwapper,
    RandomNeighborReplacer,
    RandomWordsCollapsor,
    TaMarbutaToHa,
    ZahToDad
    )


class FilesProcessor(IProcessor):
    def __init__(
            self, processes: List[IProcess],
            n_dist: int = 32
            ) -> None:
        self.processes = processes
        self.n_dist = n_dist
        self.__dist = False
        self.__cache = []

    def file_run(self, file: Union[str, Path]) -> Any:
        result = file
        for process in self.processes:
            result = process.execute(result)
        return result

    def run(
            self,
            files: List[Union[str, Path]]
            ) -> Any:
        result = list(map(self.file_run, files))
        if self.__dist is True:
            self.__cache.append(result)
            return
        return result

    def _divde(self, data: List[Any]):
        items_per_div = len(data) // self.n_dist
        divs = []
        for i in range(items_per_div):
            start = i * items_per_div
            end = (i + 1) * items_per_div
            if i == (items_per_div - 1):
                end = len(divs)
            divs.append(data[start: end])
        return divs

    def dist_run(
            self,
            files: List[Union[str, Path]]
            ) -> Any:
        self.__dist = True
        self.__cache = []
        divs = self._divde(files)
        threads = []
        for div in divs:
            t = Thread(target=self.run, args=(div,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.__dist = False
        results = []
        for item in self.__cache:
            results.extend(item)
        self.__cache = []
        return results


class TextDistorter(IProcessor):
    def __init__(
            self, ratio: float, processes: List[IProcess]
            ) -> None:
        super().__init__()
        self.ratio = ratio
        self.processes = processes

    def run(self, line: str) -> str:
        length = len(line)
        n = int(self.ratio * length)
        for _ in range(n):
            line = random.choice(self.processes).execute(line)
        return line

    def dist_run(self):
        # TODO
        pass


class TextProcessor(IProcessor):
    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__()
        self.processes = processes

    def  run(self, sentence: str):
        for process in self.processes:
            sentence = process.execute(sentence)
        return sentence

    def dist_run(self, sentence: str) -> str:
        return self.run(sentence)


class RealWorldDistorter(IProcessor):
    """Applies real-world spelling habits, then random typos.

    Real writers apply a spelling habit consistently across a sentence, so
    each habit is switched on per sentence with probability habit_prob and
    then applied to its matches (see the habit's own probability). The typo
    ratio is drawn per sentence from typo_ratios: a 0 entry keeps a share of
    typo-free sentences, the others keep the typo volume of distorted_0.1.
    """
    def __init__(
            self,
            habits: List[IProcess],
            habit_prob: float,
            typo_ratios: List[float],
            typo_processes: List[IProcess]
            ) -> None:
        super().__init__()
        self.habits = habits
        self.habit_prob = habit_prob
        self.typo_distorters = [
            TextDistorter(ratio, typo_processes) for ratio in typo_ratios
        ]

    def apply_habits(self, line: str) -> str:
        for habit in self.habits:
            if random.random() < self.habit_prob:
                line = habit.execute(line)
        return line

    def run(self, line: str) -> str:
        line = self.apply_habits(line)
        return random.choice(self.typo_distorters).run(line)

    def dist_run(self, line: str) -> str:
        return self.run(line)


def get_typo_processes() -> List[IProcess]:
    return [
        RandomCharsInjector(constants.VALID_CHARS),
        RandomCharsSwapper(),
        RandomCharRemover(),
        RandomWordsCollapsor(),
        RandomNeighborReplacer(
            constants.KEYBOARD_KEYS, constants.KEYBOARD_BLANK
            )
    ]


def get_habits(apply_prob: float) -> List[IProcess]:
    return [
        HamzaAlefDropper(apply_prob),
        TaMarbutaToHa(apply_prob),
        AlefMaqsuraToYa(apply_prob),
        HamzaSeatDropper(apply_prob),
        ZahToDad(apply_prob),
        AlefFariqaDropper(apply_prob)
    ]


def get_text_distorter(ratio):
    return TextDistorter(
        ratio=ratio,
        processes=get_typo_processes()
    )


def get_real_world_distorter(
        habit_prob: float = 0.5,
        apply_prob: float = 0.9,
        typo_ratios: List[float] = (0.0, 0.1, 0.15, 0.15)
        ) -> RealWorldDistorter:
    return RealWorldDistorter(
        habits=get_habits(apply_prob),
        habit_prob=habit_prob,
        typo_ratios=list(typo_ratios),
        typo_processes=get_typo_processes()
    )
