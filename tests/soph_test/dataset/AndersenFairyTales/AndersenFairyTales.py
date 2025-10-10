import re
from _io import TextIOWrapper
import os


class DatasetAndersen:
    def __init__(self, pth):
        with open(pth, 'r') as f:
            self._contents = [str(item).upper() for item in self._get_contents(f)]
            self._chapter_range = self._get_range(f, self._contents)
            self._chapter_map = self._get_chapter_map(f, self._chapter_range)

    def get_contents(self):
        return self._contents

    def get_chapter(self, name):
        return f'# {name}\n' + self._chapter_map[name]

    def _get_contents(self, f:TextIOWrapper) -> list:
        f.seek(0)
        conts = []
        cont_start = False
        for idx, line in enumerate(f):
            l_trip = line.rstrip()
            if cont_start:
                if re.match(r'\*+', line):
                    break
                if len(l_trip) > 0:
                    conts.append(l_trip)
            elif l_trip == 'CONTENTS':
                cont_start = True
        return conts

    def _get_range(self, f:TextIOWrapper, conts:list) -> dict:
        f.seek(0)
        titles = set(conts)
        seqs = []
        idxs = []
        finish_line = -1
        for idx, line in enumerate(f):
            l_trip = line.rstrip()
            if l_trip in titles:
                seqs.append(l_trip)
                idxs.append(idx)
                titles.remove(l_trip)
            elif re.match(r'\*+END\*+', l_trip):
                finish_line = idx
                break
        assert len(titles) == 0, f"Some chapters can not found: {titles}"
        range_map = {cont: [start, end] for cont, (start, end) in zip(seqs, zip(idxs, idxs[1:] + [finish_line]))}
        return range_map
    
    def _get_chapter_map(self, f:TextIOWrapper, cha_range:dict) -> dict:
        f.seek(0)
        lines = f.readlines()
        chapter_map = {}
        for k, (start, end) in cha_range.items():
            chapter_map[k] = ''.join(lines[start+1: end]).strip()
        return chapter_map
