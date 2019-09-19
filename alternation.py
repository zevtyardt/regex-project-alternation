#!/data/data/com.termux/files/usr/bin/python3

"""
alternation: generates one regex from a list of strings

@author val (alviandtm@gmail.com)
@date   9.8.2019

"""
import collections
import shlex
import copy
import itertools

DEBUG = 0

def debug(name, value):
    '''
    debugging -> debug(*args)
    '''
    if DEBUG:
        print("[self.{}]: {}".format(name, value))

_special_chars_map = {i: '\\' + chr(i) for i in b'()[]{}?*+-|^$\\.&~# \t\n\r\v\f'}

def _escape(pattern):
    if isinstance(pattern, str):
        return pattern.translate(_special_chars_map)
    else:
        pattern = str(pattern, 'latin1')
        return pattern.translate(_special_chars_map).encode('latin1')

def escape(s):
    """
    Escape special characters in a string.
    """
    return _escape(s).replace("\ ", " ")

class StringSet:
    '''
    a set of strings
    instead of using a list of strings everywhere, condense duplicates and preserve additional
    statistics so we can use less memory, avoid repetitive calculations and make better decisions
    '''

    def __init__(self, strings=None):
        self.strings = collections.Counter(shlex.split(strings))

    def __iter__(self):
        return iter(list(self.strings.keys()))


class Rgxg:
    def wrapper(self, strings):
        t = self.Trie(strings)
        return self.build(t)

    @classmethod
    def build(self, t):
        '''
        Directed Acyclic Word Graph
        like a Trie, but we condense substrings and share substrings/suffixes where possible
        ref: https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton
        '''
        made = {}
        for k, v in list(t.items()):
            v = Rgxg.build(v)
            # merge substrings
            if k and len(v) == 1 and '' not in v:
                k2, v2 = list(v.items())[0]
                made[k + k2] = v2
            else:
                made[k] = v
        return made

    @classmethod
    def Trie(self, strings):
        '''
        Trie
        an n-ary string character tree
        ref: https://en.wikipedia.org/wiki/Trie
        '''
        root = {}
        for word in strings:
            d = root
            for token in self.tokenize(word):
                token = escape(token)
                d = d.setdefault(token, {})
            d[''] = {}
        return root

    @classmethod
    def tokenize(self, word):
        return iter(word)


class Serialize:
    def __init__(self, dawg):
        self.ESCAPE_CHARS = ["#", "$", "&", "(", ")", "*",
                             "+", "-", ".", "?", "[", "\\",
                             "]", "^", "{", "|", "}", "~"]
        self.dawg = dawg

    @property
    def serialize(self):
        return self.serialize_regex(self.dawg)

    def serialize_regex(self, d, level=0):
        debug("serialize_regex:d", d)
        debug("serialize_regex:level", level)
        if d and self.is_char_class(d):
            s = self.as_char_class(list(d.keys()))
        elif d and self.all_suffixes_identical(d):
            v = list(d.values())[0]
            if self.all_len1(d):
                s = self.as_charclass(list(d.keys()))
            elif self.is_optional_char_class(d):
                s = self.as_opt_charclass(list(d.keys()))
            elif self.is_optional(d):
                s = self.as_optional_group(list(d.keys()))
            else:
                s = self.as_group(list(d.keys()))
            s += self.serialize_regex(v, level=level + 1)
        elif self.is_optional_char_class(d):
            s = self.as_opt_charclass(list(d.keys()))
        elif self.is_optional(d):
            s = self.opt_group(escape(sorted(list(d.keys()))[1])) + '?'
        else:
            bysuff = self.suffixes(d)
            if len(bysuff) < len(d):
                suffixed = [self.repr_keys(k, do_group=(level > 0)) +
                            self.serialize_regex(v, level=level + 1)
                            for v, k in bysuff]
                s = self.group(suffixed)
            else:
                grouped = [k + (self.serialize_regex(v, level=level + 1) if v else '')
                           for k, v in sorted(d.items())]
                s = self.group(grouped)
        return s


    # -- util --------------------
    def repr_keys(self, l, do_group=True):
        if self.all_len1(l):
            return self.as_charclass(l)
        if self.all_len01(l):
            return self.as_opt_charclass(l)
        return self.as_group(l, do_group=do_group)

    def suffixes(self, d):
        return sorted(((k, [a for a, _ in v])
                       for k, v in itertools.groupby(sorted(list(d.items()),
                                                            key=lambda x: repr(self.emptyish(x[0]))),
                                                     key=lambda x: self.emptyish(x[1]))),
                       key=lambda x: (repr(x[1]), repr(x[0])))

    def emptyish(self, x):
        if not x or x == {'': {}}:
            return {}
        return x

    def group(self, strings, do_group=True):
        debug("group:strings", strings)
        if self.all_one_backslash(strings):
            strings = self.removing_triple_backslash(strings)
            debug("group:removing_triple_backslash(strings)", strings)
        if self.is_optional_strings(strings):
            s = self.as_optional_group(strings)
            debug("is_optional_strings:group:strings", s)
            return s
        debug("group:strings:before", strings)
        s = self._subgroup(strings)
        debug("group:join(strings):before", s)
        if do_group and (len(strings) > 1 or ('|' in s and '(?:' not in s)):
            s = ('(?:' if len(strings) > 1 else '(') + s + ')'
        debug("group:join(strings):after", s)
        return s

    def _subgroup(self, l):
        for func in (self.condense_len1, self.condense_prefix):
            l = func(l)
        return '|'.join(l)

    def opt_group(self, s):
        debug("opt_group:s:before", s)
        if len(s) - s.count('\\') > 1:
            s = ('(?:' if len(s.split()) > 1 else '(') + s + ')'
        debug("opt_group:s:after", s)
        return s

    def removing_triple_backslash(self, l):
        return [x.replace("\\\\\\", "\\") for x in l]

    def reverse_string(self, s):
        if isinstance(s, list):
            return [self.reverse_string(k) for k in s]
        return s[::-1]

    # -- condense and fix --------
    def fix_spacing(self, l):
        if not isinstance(l, list):
            l = [l]
        debug("fix_spacing:l:before", l)
        for index, string in enumerate(l):
            c, space = [], 0
            for index2, char in enumerate(string, start=1):
                isspace, islast = char.isspace(), index2 == len(string)
                if islast or not isspace:
                    if space > 0:
                        space_char = (" {%s}" % (space + 1)) if space > 0 else (" " * space)
                        c.append(space_char)
                        space = 0
                    if not isspace:
                        c.append(char)
                elif isspace:
                    space += 1
            l[index] = "".join(c)
        debug("fix_spacing:l:after", l)
        if len(l) > 1:
            return sorted(l)
        return l[0]

    def condense_prefix(self, l):
        prefixes = {}
        for char in sorted(l):
            first_l, prefix = char[:1], char[1:] if len(char) > 1 else None
            if first_l.isdigit() and prefix:
                if not prefixes.get(prefix):
                    prefixes.update({prefix: []})
                prefixes[prefix].append(first_l)
                l.remove(char)
        debug("condense_prefix:prefixes", prefixes)
        for pre in prefixes:
            digits = prefixes[pre]
            s = "[{}]".format(self.condense_range(digits)) if len(digits) > 1 else digits[0]
            l.insert(0, "{}{}".format(s, pre))
        return sorted(l)

    def condense_len1(self, l):
        chr2group = []
        for char in sorted(l):
            if len(char) == 1:
                chr2group.append(char)
                l.remove(char)
        debug("condense_len1:chr2group", chr2group)
        if len(chr2group) > 0:
            s = self.condense_range(chr2group)
            if len(chr2group) > 1:
                s = "[{}]".format(s)
            l.insert(0, s)
        return sorted(l)

    def condense_range(self, chars):
        chars = sorted([c for c in chars if c])
        debug("condense_range:chars", chars)
        l = []
        while chars:
            i = 1
            while i < len(chars):
                if chars[i] != chr(ord(chars[i - 1]) + 1):
                    break
                i += 1
            if i <= 1:
                l.append(str(chars[0]))
            elif i == 2:
                l.append('{}{}'.format(chars[0], chars[1]))
            else:
                l.append('{}-{}'.format(chars[0], chars[i - 1]))
            if i == len(chars):
                chars = []
            else:
                del chars[:i]
        debug("condense_range:result", l)
        return ''.join(l)

    # -- all ---------------------
    def all_space(self, s):
        return all(x.isspace() for x in s)

    def all_one_backslash(self, l):
        for i in l:
            if "\\\\\\" in i:
                return True
        return False

    def all_escape_char(self, l):
        return all(x not in self.ESCAPE_CHARS for x in l)

    def all_digits(self, l):
        return all(x.isdigit() for x in l)

    def all_len1(self, l):
        return all(len(k) == 1 for k in l)

    def all_len01(self, l):
        return set(map(len, l)) == {0, 1}

    def all_values_not(self, d):
        return all(not v or v == {'': {}} for v in list(d.values()))

    def all_suffixes_identical(self, d):
        vals = list(d.values())
        return len(vals) > 1 and len(set(map(str, vals))) == 1

    # -- is ----------------------
    def is_repr(self, s):
        return self.all_space(s) or s.startswith(" ") or s.endswith(" ")

    def is_optional_char_class(self, d):
        return (self.all_len01(list(d.keys())) and
                self.all_values_not(d))

    def is_char_class(self, d):
        return (self.all_len1(list(d.keys())) and
                self.all_values_not(d))

    def is_optional_strings(self, strings):
        return not all(s for s in strings)

    def is_optional(self, d):
        if len(d) == 2:
            items = sorted(list(d.items()))
            return (not items[0][0] and (
                    not items[1][1] or items[1][1] == {'': {}}))
        return False

    def is_unescape_char_in_string(self, strings):
        for index, char in enumerate(strings):
            if char in self.ESCAPE_CHARS:
                if index == 0:
                    if len(strings) > 1:
                        if char == "\\" and strings[index + 1] in self.ESCAPE_CHARS:
                            continue
                    return True
                else:
                    if not strings[index - 1].endswith("\\"):
                        if index < len(strings):
                            if char == "\\" and strings[index + 1] in self.ESCAPE_CHARS:
                                continue
                        return True
        return False

    # -- as ----------------------
    def as_repr(self, s):
        return repr(s).replace("\\\\", "\\")

    def as_group(self, l, do_group=True):
        l = sorted(l)
        debug("as_group:sorted(l)", l)
        suffix, dogroup = self._as_group_subfind_suffix_and_dogroup(l, do_group)
        if suffix:
            lensuff = len(suffix)
            prefixes = [x[:-lensuff] for x in l]
            debug("as_group:prefixes", prefixes)
            if self.all_len1(prefixes):
                s = self.as_char_class(prefixes)
            else:
                s = self.group(prefixes)
            s += suffix
        else:
            s = self.group(l, do_group=dogroup)
        debug("as_group:result", s)
        return s

    def _as_group_subfind_suffix_and_dogroup(self, l, do_group):
        dogroup = suffix = self.longest_suffix(l) if len(l) > 1 else ''
        debug("as_group:suffix:before", suffix)
        suffix = suffix if not self.is_unescape_char_in_string(suffix) else ""
        debug("as_group:suffix:after", suffix)
        dogroup = (True if suffix else False) if dogroup != "" else do_group
        debug("as_group:dogroup", dogroup)
        return suffix, dogroup

    def as_optional_group(self, strings):
        strings = sorted(strings)
        debug("as_optional_group:sorted(strings)", strings)
        assert strings[0] == ''
        j = strings[1:]
        if not j:
            return ''
        debug("as_optional_group:j", j)
        if self.all_digits(j) and self.all_len1(j):
            s = self.condense_range(j)
            if len(s) > 1:
                s = "[{}]".format(s)
            debug("as_optional_group:all_digits(j):condense_range", s)
        else:
            if len(j) > 1:
                j = [self.as_group(j)]
            s = '|'.join(j)
            debug("as_optional_group:join(j):before", s)
            if len(j) > 1 or len(j[0]) > 1 or s.endswith('?') or '|' in s or '(?:' in s:
                if not s.startswith("(?:"):
                    s = ('(?:' if len(j) > 1 else '(') + s + ')'
            debug("as_optional_group:join(j):after", s)
        result = s + "?"
        debug("as_optional_group:result", result)
        return result

    def as_char_class(self, strings):
        debug("as_char_class:strings", strings)
        s = ''.join(sorted(strings))
        debug("as_char_class:s:before", s)
        if len(s) > 1:
            s = "[{}]".format(self.condense_range(s))
            debug("as_char_class:condense_range", s)
        debug("as_char_class:s:after", s)
        return s

    def as_opt_charclass(self, l):
        debug("as_opt_charclass:l", l)
        s = self.condense_range(l)
        debug("as_opt_charclass:condense_range", s)
        if len(l) > 2:
            s = '[' + s + ']'
            debug("as_opt_charclass:len(l) > 2", s)
        else:
            s = escape(s)
            debug("as_opt_charclass:escape", s)
        result = s + "?"
        debug("condense_range:result", result)
        return result

    def as_charclass(self, l):
        debug("as_charclass:l", l)
        s = self.condense_range(l)
        debug("as_charclass:condense_range", s)
        if len(l) > 1:
            s = '[' + s + ']'
            debug("as_charclass:len(l) > 1", s)
        return s

    def as_unicode(self, s):
        return bytes(s, "utf-8").decode("unicode_escape")

    # -- longest -----------------
    def longest_prefix_2strings(self, x, y, longest):
        length = min(min(len(x), len(y)), longest)
        for i in range(1, length + 1):
            if x[:i] != y[:i]:
                return i - 1
        return length

    def longest_suffix(self, strings):
        return self.longest_prefix([s[::-1] for s in copy.copy(strings)])

    def longest_prefix(self, strings):
        if not strings:
            return ''
        prefix = strings[0]
        longest = min(len(s) for s in strings)
        for i in range(1, len(strings)):
            longest = self.longest_prefix_2strings(prefix, strings[i], longest)
            if longest == 0:
                return ''
        return prefix[:longest][::-1]

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.dawg)

    def __dict__(self):
        return self.dawg

def match(s, verbose=False):
    '''
    convenience wrapper for generating one regex from a list of strings
    '''
    global DEBUG
    DEBUG = verbose or DEBUG
    debug("generic:s", s)
    if not isinstance(s, list):
        s = StringSet(s)
        debug("generic:not isinstance(s, list)", s)
    trie = Rgxg.Trie(s)
    dawg = Rgxg.build(trie)
    serial = Serialize(dawg)
    return (lambda x: serial.as_repr(x) if serial.is_repr(x) else x)(serial.serialize)

if __name__ == "__main__":
    import sys
    debug("generic:debug", "True")
    final = match(sys.argv[1:], verbose=True)
    if DEBUG:
        print("-" * 25)
        debug("generic:final", final)
    else:
        sys.exit(final)
