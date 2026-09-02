"""LeetCode medium/hard eval bank.

Each problem includes official examples plus extra public edge cases. Expected
outputs are checked against the local reference implementation at import time
when ``validate_problem_bank`` is called.
"""

from __future__ import annotations

import random
from typing import Any, Callable

PINNED_IDS = ("lc3", "lc42")


def _prompt_block(
    number: int,
    title: str,
    description: str,
    examples: str,
    prelude: str,
    function_header: str,
    think: bool,
) -> str:
    suffix = "    <think>\n" if think else "    "
    return (
        f'"""\n'
        f"LeetCode {number}: {title}\n\n"
        f"{description}\n\n"
        f"{examples}\n"
        f'"""\n'
        f"{prelude}\n\n"
        f"{function_header}\n"
        f"{suffix}"
    )


def make_problem(
    *,
    number: int,
    title: str,
    difficulty: str,
    slug: str,
    method_name: str,
    function_header: str,
    description: str,
    examples: str,
    test_cases: list[dict[str, Any]],
    reference: Callable[..., Any],
    compare: str = "eq",
    prelude: str = "from typing import List",
) -> dict[str, Any]:
    return {
        "id": f"lc{number}",
        "number": number,
        "title": f"LeetCode {number}: {title}",
        "short_title": title,
        "difficulty": difficulty,
        "url": f"https://leetcode.com/problems/{slug}/",
        "method_name": method_name,
        "function_header": function_header,
        "description": description,
        "examples": examples,
        "prelude": prelude,
        "compare": compare,
        "test_cases": test_cases,
        "reference": reference,
        "prompt_base": _prompt_block(
            number, title, description, examples, prelude, function_header, False
        ),
        "prompt_think": _prompt_block(
            number, title, description, examples, prelude, function_header, True
        ),
    }


def _ref_length_of_longest_substring(s: str) -> int:
    last = {}
    start = 0
    best = 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best


def _ref_max_area(height: list[int]) -> int:
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        best = max(best, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best


def _ref_three_sum(nums: list[int]) -> list[list[int]]:
    nums = sorted(nums)
    out: list[list[int]] = []
    n = len(nums)
    for i in range(n):
        if i and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total < 0:
                lo += 1
            elif total > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
    return out


def _ref_max_sub_array(nums: list[int]) -> int:
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def _ref_merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _ref_num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0

    def dfs(r: int, c: int) -> None:
        stack = [(r, c)]
        seen[r][c] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and grid[nr][nc] == "1":
                    seen[nr][nc] = True
                    stack.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and not seen[r][c]:
                count += 1
                dfs(r, c)
    return count


def _ref_product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):
        out[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out


def _ref_coin_change(coins: list[int], amount: int) -> int:
    inf = amount + 1
    dp = [0] + [inf] * amount
    for x in range(1, amount + 1):
        for coin in coins:
            if coin <= x:
                dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1


def _ref_find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    merged = sorted(nums1 + nums2)
    n = len(merged)
    mid = n // 2
    if n % 2:
        return float(merged[mid])
    return (merged[mid - 1] + merged[mid]) / 2.0


def _ref_longest_valid_parentheses(s: str) -> int:
    stack = [-1]
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                best = max(best, i - stack[-1])
    return best


def _ref_first_missing_positive(nums: list[int]) -> int:
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i, x in enumerate(nums):
        if x != i + 1:
            return i + 1
    return n + 1


def _ref_trap(height: list[int]) -> int:
    if not height:
        return 0
    lo, hi = 0, len(height) - 1
    left_max, right_max = height[lo], height[hi]
    water = 0
    while lo < hi:
        if left_max <= right_max:
            lo += 1
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
        else:
            hi -= 1
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
    return water


def _ref_min_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if word1[i - 1] == word2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def _ref_min_window(s: str, t: str) -> str:
    if not t or len(t) > len(s):
        return ""
    need: dict[str, int] = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    missing = len(t)
    best_len = len(s) + 1
    best = ""
    lo = 0
    for hi, ch in enumerate(s):
        if ch in need:
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1
        while missing == 0:
            if hi - lo + 1 < best_len:
                best_len = hi - lo + 1
                best = s[lo : hi + 1]
            left = s[lo]
            if left in need:
                need[left] += 1
                if need[left] > 0:
                    missing += 1
            lo += 1
    return best


def _ref_largest_rectangle_area(heights: list[int]) -> int:
    stack: list[int] = []
    best = 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best


def _ref_max_sliding_window(nums: list[int], k: int) -> list[int]:
    from collections import deque

    dq: deque[int] = deque()
    out: list[int] = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


PROBLEM_BANK: list[dict[str, Any]] = [
    make_problem(
        number=3,
        title="Longest Substring Without Repeating Characters",
        difficulty="Medium",
        slug="longest-substring-without-repeating-characters",
        method_name="lengthOfLongestSubstring",
        function_header="def lengthOfLongestSubstring(s: str) -> int:",
        description="Given a string s, find the length of the longest substring without duplicate characters.",
        examples=(
            'Example 1:\n'
            'Input: s = "abcabcbb"\n'
            "Output: 3\n"
            'Explanation: The answer is "abc", with the length of 3.\n\n'
            "Example 2:\n"
            'Input: s = "bbbbb"\n'
            "Output: 1\n"
            'Explanation: The answer is "b", with the length of 1.\n\n'
            "Example 3:\n"
            'Input: s = "pwwkew"\n'
            "Output: 3\n"
            'Explanation: The answer is "wke", with the length of 3.\n'
            'Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.'
        ),
        prelude="",
        test_cases=[
            {"args": ["abcabcbb"], "expected": 3},
            {"args": ["bbbbb"], "expected": 1},
            {"args": ["pwwkew"], "expected": 3},
            {"args": [""], "expected": 0},
            {"args": [" "], "expected": 1},
            {"args": ["au"], "expected": 2},
            {"args": ["dvdf"], "expected": 3},
            {"args": ["abba"], "expected": 2},
            {"args": ["tmmzuxt"], "expected": 5},
            {"args": ["a"], "expected": 1},
            {"args": ["abcdef"], "expected": 6},
            {"args": ["aab"], "expected": 2},
            {"args": ["cdd"], "expected": 2},
            {"args": ["bbtabl"], "expected": 4},
            {"args": ["anviaj"], "expected": 5},
            {"args": ["ohvhjdml"], "expected": 6},
            {"args": ["abccba"], "expected": 3},
        ],
        reference=_ref_length_of_longest_substring,
    ),
    make_problem(
        number=4,
        title="Median of Two Sorted Arrays",
        difficulty="Hard",
        slug="median-of-two-sorted-arrays",
        method_name="findMedianSortedArrays",
        function_header="def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:",
        description=(
            "Given two sorted arrays nums1 and nums2 of size m and n respectively, "
            "return the median of the two sorted arrays."
        ),
        examples=(
            "Example 1:\n"
            "Input: nums1 = [1,3], nums2 = [2]\n"
            "Output: 2.00000\n"
            "Explanation: merged array = [1,2,3] and median is 2.\n\n"
            "Example 2:\n"
            "Input: nums1 = [1,2], nums2 = [3,4]\n"
            "Output: 2.50000\n"
            "Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5."
        ),
        compare="float",
        test_cases=[
            {"args": [[1, 3], [2]], "expected": 2.0},
            {"args": [[1, 2], [3, 4]], "expected": 2.5},
            {"args": [[0, 0], [0, 0]], "expected": 0.0},
            {"args": [[], [1]], "expected": 1.0},
            {"args": [[2], []], "expected": 2.0},
            {"args": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], "expected": 5.5},
            {"args": [[1, 2], [1, 2, 3]], "expected": 2.0},
            {"args": [[1], [1]], "expected": 1.0},
            {"args": [[1, 3, 5, 7], [2, 4, 6, 8]], "expected": 4.5},
            {"args": [[], [2, 3]], "expected": 2.5},
        ],
        reference=_ref_find_median_sorted_arrays,
    ),
    make_problem(
        number=11,
        title="Container With Most Water",
        difficulty="Medium",
        slug="container-with-most-water",
        method_name="maxArea",
        function_header="def maxArea(height: List[int]) -> int:",
        description=(
            "You are given an integer array height of length n. There are n vertical "
            "lines drawn such that the two endpoints of the ith line are (i, 0) and "
            "(i, height[i]). Find two lines that together with the x-axis form a "
            "container such that the container contains the most water. Return the "
            "maximum amount of water a container can store."
        ),
        examples=(
            "Example 1:\n"
            "Input: height = [1,8,6,2,5,4,8,3,7]\n"
            "Output: 49\n"
            "Explanation: The above vertical lines are represented by array "
            "[1,8,6,2,5,4,8,3,7]. The max area of water (blue section) the container "
            "can contain is 49.\n\n"
            "Example 2:\n"
            "Input: height = [1,1]\n"
            "Output: 1"
        ),
        test_cases=[
            {"args": [[1, 8, 6, 2, 5, 4, 8, 3, 7]], "expected": 49},
            {"args": [[1, 1]], "expected": 1},
            {"args": [[1, 2, 1]], "expected": 2},
            {"args": [[4, 3, 2, 1, 4]], "expected": 16},
            {"args": [[1, 2, 4, 3]], "expected": 4},
            {"args": [[2, 3, 4, 5, 18, 17, 6]], "expected": 17},
            {"args": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], "expected": 25},
            {"args": [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], "expected": 25},
            {"args": [[5, 5, 5, 5]], "expected": 15},
        ],
        reference=_ref_max_area,
    ),
    make_problem(
        number=15,
        title="3Sum",
        difficulty="Medium",
        slug="3sum",
        method_name="threeSum",
        function_header="def threeSum(nums: List[int]) -> List[List[int]]:",
        description=(
            "Given an integer array nums, return all the triplets [nums[i], nums[j], "
            "nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + "
            "nums[k] == 0. The solution set must not contain duplicate triplets."
        ),
        examples=(
            "Example 1:\n"
            "Input: nums = [-1,0,1,2,-1,-4]\n"
            "Output: [[-1,-1,2],[-1,0,1]]\n\n"
            "Example 2:\n"
            "Input: nums = [0,1,1]\n"
            "Output: []\n\n"
            "Example 3:\n"
            "Input: nums = [0,0,0]\n"
            "Output: [[0,0,0]]"
        ),
        compare="sorted_triplets",
        test_cases=[
            {"args": [[-1, 0, 1, 2, -1, -4]], "expected": [[-1, -1, 2], [-1, 0, 1]]},
            {"args": [[0, 1, 1]], "expected": []},
            {"args": [[0, 0, 0]], "expected": [[0, 0, 0]]},
            {"args": [[0, 0, 0, 0]], "expected": [[0, 0, 0]]},
            {"args": [[-2, 0, 1, 1, 2]], "expected": [[-2, 0, 2], [-2, 1, 1]]},
            {"args": [[-1, 0, 1]], "expected": [[-1, 0, 1]]},
            {"args": [[1, 2, -2, -1]], "expected": []},
            {"args": [[-4, -2, -2, -2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6]], "expected": [[-4, -2, 6], [-4, 0, 4], [-4, 1, 3], [-4, 2, 2], [-2, -2, 4], [-2, 0, 2]]},
            {"args": [[3, 0, -2, -1, 1, 2]], "expected": [[-2, -1, 3], [-2, 0, 2], [-1, 0, 1]]},
            {"args": [[-1, -1, -1, 2, 2]], "expected": [[-1, -1, 2]]},
        ],
        reference=_ref_three_sum,
    ),
    make_problem(
        number=32,
        title="Longest Valid Parentheses",
        difficulty="Hard",
        slug="longest-valid-parentheses",
        method_name="longestValidParentheses",
        function_header="def longestValidParentheses(s: str) -> int:",
        description="Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses substring.",
        examples=(
            "Example 1:\n"
            'Input: s = "(()"\n'
            "Output: 2\n"
            'Explanation: The longest valid parentheses substring is "()".\n\n'
            "Example 2:\n"
            'Input: s = ")()())"\n'
            "Output: 4\n"
            'Explanation: The longest valid parentheses substring is "()()".\n\n'
            "Example 3:\n"
            'Input: s = ""\n'
            "Output: 0"
        ),
        prelude="",
        test_cases=[
            {"args": ["(()"], "expected": 2},
            {"args": [")()())"], "expected": 4},
            {"args": [""], "expected": 0},
            {"args": ["()"], "expected": 2},
            {"args": ["()(()"], "expected": 2},
            {"args": ["()(())"], "expected": 6},
            {"args": ["((((("], "expected": 0},
            {"args": [")))))"], "expected": 0},
            {"args": ["()()()"], "expected": 6},
            {"args": ["(()())"], "expected": 6},
            {"args": ["(()(((()"], "expected": 2},
            {"args": [")(()())("], "expected": 6},
        ],
        reference=_ref_longest_valid_parentheses,
    ),
    make_problem(
        number=41,
        title="First Missing Positive",
        difficulty="Hard",
        slug="first-missing-positive",
        method_name="firstMissingPositive",
        function_header="def firstMissingPositive(nums: List[int]) -> int:",
        description="Given an unsorted integer array nums. Return the smallest positive integer that is not present in nums. You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.",
        examples=(
            "Example 1:\n"
            "Input: nums = [1,2,0]\n"
            "Output: 3\n\n"
            "Example 2:\n"
            "Input: nums = [3,4,-1,1]\n"
            "Output: 2\n\n"
            "Example 3:\n"
            "Input: nums = [7,8,9,11,12]\n"
            "Output: 1"
        ),
        test_cases=[
            {"args": [[1, 2, 0]], "expected": 3},
            {"args": [[3, 4, -1, 1]], "expected": 2},
            {"args": [[7, 8, 9, 11, 12]], "expected": 1},
            {"args": [[1]], "expected": 2},
            {"args": [[1, 1]], "expected": 2},
            {"args": [[2, 1]], "expected": 3},
            {"args": [[-1, -2]], "expected": 1},
            {"args": [[1, 2, 3]], "expected": 4},
            {"args": [[2, 2]], "expected": 1},
            {"args": [[0]], "expected": 1},
            {"args": [[1, 2, 4]], "expected": 3},
        ],
        reference=_ref_first_missing_positive,
    ),
    make_problem(
        number=42,
        title="Trapping Rain Water",
        difficulty="Hard",
        slug="trapping-rain-water",
        method_name="trap",
        function_header="def trap(height: List[int]) -> int:",
        description="Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
        examples=(
            "Example 1:\n"
            "Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]\n"
            "Output: 6\n"
            "Explanation: The above elevation map (black section) is represented by "
            "array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water "
            "(blue section) are being trapped.\n\n"
            "Example 2:\n"
            "Input: height = [4,2,0,3,2,5]\n"
            "Output: 9"
        ),
        test_cases=[
            {"args": [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]], "expected": 6},
            {"args": [[4, 2, 0, 3, 2, 5]], "expected": 9},
            {"args": [[]], "expected": 0},
            {"args": [[1]], "expected": 0},
            {"args": [[1, 2]], "expected": 0},
            {"args": [[2, 0, 2]], "expected": 2},
            {"args": [[3, 0, 0, 2, 0, 4]], "expected": 10},
            {"args": [[0, 0, 0, 0]], "expected": 0},
            {"args": [[5, 4, 1, 2]], "expected": 1},
            {"args": [[5, 2, 1, 2, 1, 5]], "expected": 14},
            {"args": [[0, 2, 0]], "expected": 0},
            {"args": [[4, 2, 3]], "expected": 1},
            {"args": [[2, 1, 0, 2]], "expected": 3},
            {"args": [[10, 0, 10]], "expected": 10},
            {"args": [[1, 2, 3, 4, 5]], "expected": 0},
            {"args": [[5, 4, 3, 2, 1]], "expected": 0},
            {"args": [[6, 4, 2, 0, 3, 2, 0, 3, 1, 4, 5, 3, 2, 7, 5, 3, 0, 1, 2, 1, 3, 4, 6, 8, 1, 3]], "expected": 83},
        ],
        reference=_ref_trap,
    ),
    make_problem(
        number=53,
        title="Maximum Subarray",
        difficulty="Medium",
        slug="maximum-subarray",
        method_name="maxSubArray",
        function_header="def maxSubArray(nums: List[int]) -> int:",
        description="Given an integer array nums, find the subarray with the largest sum, and return its sum.",
        examples=(
            "Example 1:\n"
            "Input: nums = [-2,1,-3,4,-1,2,1,-5,4]\n"
            "Output: 6\n"
            "Explanation: The subarray [4,-1,2,1] has the largest sum 6.\n\n"
            "Example 2:\n"
            "Input: nums = [1]\n"
            "Output: 1\n\n"
            "Example 3:\n"
            "Input: nums = [5,4,-1,7,8]\n"
            "Output: 23"
        ),
        test_cases=[
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1]], "expected": 1},
            {"args": [[5, 4, -1, 7, 8]], "expected": 23},
            {"args": [[-1]], "expected": -1},
            {"args": [[-2, -1]], "expected": -1},
            {"args": [[-2, 1]], "expected": 1},
            {"args": [[0, -3, 1, 1]], "expected": 2},
            {"args": [[-5, -4, -3]], "expected": -3},
            {"args": [[8, -19, 5, -1, 7, -5, 1]], "expected": 11},
            {"args": [[1, 2, 3, 4]], "expected": 10},
        ],
        reference=_ref_max_sub_array,
    ),
    make_problem(
        number=56,
        title="Merge Intervals",
        difficulty="Medium",
        slug="merge-intervals",
        method_name="merge",
        function_header="def merge(intervals: List[List[int]]) -> List[List[int]]:",
        description="Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.",
        examples=(
            "Example 1:\n"
            "Input: intervals = [[1,3],[2,6],[8,10],[15,18]]\n"
            "Output: [[1,6],[8,10],[15,18]]\n"
            "Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].\n\n"
            "Example 2:\n"
            "Input: intervals = [[1,4],[4,5]]\n"
            "Output: [[1,5]]\n"
            "Explanation: Intervals [1,4] and [4,5] are considered overlapping."
        ),
        compare="sorted_intervals",
        test_cases=[
            {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expected": [[1, 6], [8, 10], [15, 18]]},
            {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            {"args": [[[1, 4], [0, 4]]], "expected": [[0, 4]]},
            {"args": [[[1, 4], [2, 3]]], "expected": [[1, 4]]},
            {"args": [[[1, 4], [0, 0]]], "expected": [[0, 0], [1, 4]]},
            {"args": [[[1, 4], [0, 2], [3, 5]]], "expected": [[0, 5]]},
            {"args": [[[1, 3]]], "expected": [[1, 3]]},
            {"args": [[[2, 3], [4, 5], [6, 7], [8, 9], [1, 10]]], "expected": [[1, 10]]},
            {"args": [[[1, 4], [5, 6]]], "expected": [[1, 4], [5, 6]]},
        ],
        reference=_ref_merge,
    ),
    make_problem(
        number=72,
        title="Edit Distance",
        difficulty="Hard",
        slug="edit-distance",
        method_name="minDistance",
        function_header="def minDistance(word1: str, word2: str) -> int:",
        description="Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. You have three operations: insert a character, delete a character, or replace a character.",
        examples=(
            "Example 1:\n"
            'Input: word1 = "horse", word2 = "ros"\n'
            "Output: 3\n"
            "Explanation:\n"
            "horse -> rorse (replace 'h' with 'r')\n"
            "rorse -> rose (delete 'r')\n"
            "rose -> ros (delete 'e')\n\n"
            "Example 2:\n"
            'Input: word1 = "intention", word2 = "execution"\n'
            "Output: 5"
        ),
        prelude="",
        test_cases=[
            {"args": ["horse", "ros"], "expected": 3},
            {"args": ["intention", "execution"], "expected": 5},
            {"args": ["", ""], "expected": 0},
            {"args": ["a", ""], "expected": 1},
            {"args": ["", "a"], "expected": 1},
            {"args": ["a", "a"], "expected": 0},
            {"args": ["ab", "bc"], "expected": 2},
            {"args": ["plasma", "altruism"], "expected": 6},
            {"args": ["sea", "eat"], "expected": 2},
            {"args": ["dinitrophenylhydrazine", "acetylphenylhydrazine"], "expected": 6},
        ],
        reference=_ref_min_distance,
    ),
    make_problem(
        number=76,
        title="Minimum Window Substring",
        difficulty="Hard",
        slug="minimum-window-substring",
        method_name="minWindow",
        function_header="def minWindow(s: str, t: str) -> str:",
        description="Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string.",
        examples=(
            "Example 1:\n"
            'Input: s = "ADOBECODEBANC", t = "ABC"\n'
            'Output: "BANC"\n'
            "Explanation: The minimum window substring \"BANC\" includes 'A', 'B', and 'C' from string t.\n\n"
            "Example 2:\n"
            'Input: s = "a", t = "a"\n'
            'Output: "a"\n\n'
            "Example 3:\n"
            'Input: s = "a", t = "aa"\n'
            'Output: ""'
        ),
        prelude="",
        test_cases=[
            {"args": ["ADOBECODEBANC", "ABC"], "expected": "BANC"},
            {"args": ["a", "a"], "expected": "a"},
            {"args": ["a", "aa"], "expected": ""},
            {"args": ["ab", "b"], "expected": "b"},
            {"args": ["aa", "aa"], "expected": "aa"},
            {"args": ["bba", "ab"], "expected": "ba"},
            {"args": ["cabwefgewcwaefgcf", "cae"], "expected": "cwae"},
            {"args": ["ab", "a"], "expected": "a"},
            {"args": ["aaflslflsldkalskaaa", "aaa"], "expected": "aaa"},
            {"args": ["abc", "d"], "expected": ""},
        ],
        reference=_ref_min_window,
    ),
    make_problem(
        number=84,
        title="Largest Rectangle in Histogram",
        difficulty="Hard",
        slug="largest-rectangle-in-histogram",
        method_name="largestRectangleArea",
        function_header="def largestRectangleArea(heights: List[int]) -> int:",
        description="Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.",
        examples=(
            "Example 1:\n"
            "Input: heights = [2,1,5,6,2,3]\n"
            "Output: 10\n"
            "Explanation: The above is a histogram where width of each bar is 1. The largest rectangle is shown in the red area, which has an area = 10 units.\n\n"
            "Example 2:\n"
            "Input: heights = [2,4]\n"
            "Output: 4"
        ),
        test_cases=[
            {"args": [[2, 1, 5, 6, 2, 3]], "expected": 10},
            {"args": [[2, 4]], "expected": 4},
            {"args": [[1]], "expected": 1},
            {"args": [[0]], "expected": 0},
            {"args": [[1, 1, 1, 1]], "expected": 4},
            {"args": [[5, 4, 3, 2, 1]], "expected": 9},
            {"args": [[1, 2, 3, 4, 5]], "expected": 9},
            {"args": [[0, 1, 0, 1, 0]], "expected": 1},
            {"args": [[6, 2, 5, 4, 5, 1, 6]], "expected": 12},
            {"args": [[2, 1, 2]], "expected": 3},
        ],
        reference=_ref_largest_rectangle_area,
    ),
    make_problem(
        number=200,
        title="Number of Islands",
        difficulty="Medium",
        slug="number-of-islands",
        method_name="numIslands",
        function_header="def numIslands(grid: List[List[str]]) -> int:",
        description='Given an m x n 2D binary grid which represents a map of \'1\'s (land) and \'0\'s (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.',
        examples=(
            "Example 1:\n"
            'Input: grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]\n'
            "Output: 1\n\n"
            "Example 2:\n"
            'Input: grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]\n'
            "Output: 3"
        ),
        test_cases=[
            {
                "args": [[["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]],
                "expected": 1,
            },
            {
                "args": [[["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]]],
                "expected": 3,
            },
            {"args": [[["1"]]], "expected": 1},
            {"args": [[["0"]]], "expected": 0},
            {"args": [[["1", "0", "1", "0", "1"]]], "expected": 3},
            {"args": [[["1"], ["0"], ["1"], ["0"], ["1"]]], "expected": 3},
            {
                "args": [[["1", "1", "1"], ["0", "1", "0"], ["1", "1", "1"]]],
                "expected": 1,
            },
            {"args": [[["0", "0"], ["0", "0"]]], "expected": 0},
        ],
        reference=_ref_num_islands,
    ),
    make_problem(
        number=238,
        title="Product of Array Except Self",
        difficulty="Medium",
        slug="product-of-array-except-self",
        method_name="productExceptSelf",
        function_header="def productExceptSelf(nums: List[int]) -> List[int]:",
        description="Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation.",
        examples=(
            "Example 1:\n"
            "Input: nums = [1,2,3,4]\n"
            "Output: [24,12,8,6]\n\n"
            "Example 2:\n"
            "Input: nums = [-1,1,0,-3,3]\n"
            "Output: [0,0,9,0,0]"
        ),
        test_cases=[
            {"args": [[1, 2, 3, 4]], "expected": [24, 12, 8, 6]},
            {"args": [[-1, 1, 0, -3, 3]], "expected": [0, 0, 9, 0, 0]},
            {"args": [[0, 0]], "expected": [0, 0]},
            {"args": [[1, 0]], "expected": [0, 1]},
            {"args": [[2, 3, 4, 5]], "expected": [60, 40, 30, 24]},
            {"args": [[1, 1, 1, 1]], "expected": [1, 1, 1, 1]},
            {"args": [[-1, -1]], "expected": [-1, -1]},
            {"args": [[0, 4, 0]], "expected": [0, 0, 0]},
        ],
        reference=_ref_product_except_self,
    ),
    make_problem(
        number=239,
        title="Sliding Window Maximum",
        difficulty="Hard",
        slug="sliding-window-maximum",
        method_name="maxSlidingWindow",
        function_header="def maxSlidingWindow(nums: List[int], k: int) -> List[int]:",
        description="You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Return the max sliding window.",
        examples=(
            "Example 1:\n"
            "Input: nums = [1,3,-1,-3,5,3,6,7], k = 3\n"
            "Output: [3,3,5,5,6,7]\n\n"
            "Example 2:\n"
            "Input: nums = [1], k = 1\n"
            "Output: [1]"
        ),
        test_cases=[
            {"args": [[1, 3, -1, -3, 5, 3, 6, 7], 3], "expected": [3, 3, 5, 5, 6, 7]},
            {"args": [[1], 1], "expected": [1]},
            {"args": [[9, 11], 2], "expected": [11]},
            {"args": [[4, -2], 2], "expected": [4]},
            {"args": [[1, 3, 1, 2, 0, 5], 3], "expected": [3, 3, 2, 5]},
            {"args": [[7, 2, 4], 2], "expected": [7, 4]},
            {"args": [[1, -1], 1], "expected": [1, -1]},
            {"args": [[10, 9, 8, 7, 6, 5], 3], "expected": [10, 9, 8, 7]},
            {"args": [[1, 2, 3, 4, 5], 5], "expected": [5]},
        ],
        reference=_ref_max_sliding_window,
    ),
    make_problem(
        number=322,
        title="Coin Change",
        difficulty="Medium",
        slug="coin-change",
        method_name="coinChange",
        function_header="def coinChange(coins: List[int], amount: int) -> int:",
        description="You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1. You may assume that you have an infinite number of each kind of coin.",
        examples=(
            "Example 1:\n"
            "Input: coins = [1,2,5], amount = 11\n"
            "Output: 3\n"
            "Explanation: 11 = 5 + 5 + 1\n\n"
            "Example 2:\n"
            "Input: coins = [2], amount = 3\n"
            "Output: -1\n\n"
            "Example 3:\n"
            "Input: coins = [1], amount = 0\n"
            "Output: 0"
        ),
        test_cases=[
            {"args": [[1, 2, 5], 11], "expected": 3},
            {"args": [[2], 3], "expected": -1},
            {"args": [[1], 0], "expected": 0},
            {"args": [[1], 1], "expected": 1},
            {"args": [[1], 2], "expected": 2},
            {"args": [[1, 3, 4], 6], "expected": 2},
            {"args": [[2, 5, 10, 1], 27], "expected": 4},
            {"args": [[186, 419, 83, 408], 6249], "expected": 20},
            {"args": [[3, 7], 5], "expected": -1},
            {"args": [[5], 10], "expected": 2},
        ],
        reference=_ref_coin_change,
    ),
]


def results_equal(got: Any, expected: Any, compare: str) -> bool:
    if compare == "float":
        try:
            return abs(float(got) - float(expected)) < 1e-5
        except (TypeError, ValueError):
            return False
    if compare == "sorted_triplets":
        try:
            got_n = sorted(tuple(sorted(t)) for t in got)
            exp_n = sorted(tuple(sorted(t)) for t in expected)
            return got_n == exp_n
        except TypeError:
            return False
    if compare == "sorted_intervals":
        try:
            return sorted(list(x) for x in got) == sorted(list(x) for x in expected)
        except TypeError:
            return False
    if isinstance(expected, list):
        try:
            if isinstance(got, tuple):
                got = list(got)
            if isinstance(got, list) and expected and isinstance(expected[0], list):
                got = [list(x) for x in got]
            return got == expected
        except TypeError:
            return False
    return got == expected


def validate_problem_bank(problems: list[dict[str, Any]] | None = None) -> None:
    problems = problems if problems is not None else PROBLEM_BANK
    for problem in problems:
        ref = problem["reference"]
        compare = problem["compare"]
        for i, case in enumerate(problem["test_cases"]):
            args = [__import__("copy").deepcopy(a) for a in case["args"]]
            got = ref(*args)
            if not results_equal(got, case["expected"], compare):
                raise AssertionError(
                    f"{problem['id']} test {i}: reference returned {got!r}, "
                    f"expected {case['expected']!r}"
                )


def sample_problems(
    n: int = 10,
    seed: int = 42,
    ids: list[str] | None = None,
    pin_original: bool = True,
) -> list[dict[str, Any]]:
    by_id = {p["id"]: p for p in PROBLEM_BANK}
    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"Unknown problem ids: {missing}. Known: {sorted(by_id)}")
        return [by_id[i] for i in ids]

    if n >= len(PROBLEM_BANK):
        return list(PROBLEM_BANK)

    rng = random.Random(seed)
    if pin_original:
        pinned = [by_id[i] for i in PINNED_IDS if i in by_id]
        rest = [p for p in PROBLEM_BANK if p["id"] not in PINNED_IDS]
        n_extra = max(0, n - len(pinned))
        extra = rng.sample(rest, k=min(n_extra, len(rest)))
        selected = pinned + extra
    else:
        selected = rng.sample(PROBLEM_BANK, k=n)
    selected.sort(key=lambda p: p["number"])
    return selected


def problem_public_dict(problem: dict[str, Any]) -> dict[str, Any]:
    skip = {"reference", "prompt_base", "prompt_think"}
    return {k: v for k, v in problem.items() if k not in skip}


if __name__ == "__main__":
    validate_problem_bank()
    print(f"Validated {len(PROBLEM_BANK)} problems.")
