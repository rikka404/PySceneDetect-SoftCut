#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2014-2024 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
""":class:`ContentDetector` compares the difference in content between adjacent frames against a
set threshold/score, which if exceeded, triggers a scene cut.

This detector is available from the command-line as the `detect-content` command.
"""

import math
import typing as ty
from dataclasses import dataclass

import cv2
import numpy

from scenedetect.common import FrameTimecode
from scenedetect.detector import FlashFilter, SceneDetector

import numpy as np
import logging

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')
with open("log.txt", "w") as f:
    pass


def _mean_pixel_distance(left: numpy.ndarray, right: numpy.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return numpy.sum(numpy.abs(left.astype(numpy.int32) - right.astype(numpy.int32))) / num_pixels


def _estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    # TODO: This equation is based on manual estimation from a few videos.
    # Create a more comprehensive test suite to optimize against.
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size


class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    The difference is calculated in the HSV color space, and compared against a set threshold to
    determine when a fast cut has occurred.
    """

    # TODO: Come up with some good weights for a new default if there is one that can pass
    # a wider variety of test cases.
    class Components(ty.NamedTuple):
        """Components that make up a frame's score, and their default values."""

        delta_hue: float = 1.0
        """Difference between pixel hue values of adjacent frames."""
        delta_sat: float = 1.0
        """Difference between pixel saturation values of adjacent frames."""
        delta_lum: float = 1.0
        """Difference between pixel luma (brightness) values of adjacent frames."""
        delta_edges: float = 0.0
        """Difference between calculated edges of adjacent frames.

        Edge differences are typically larger than the other components, so the detection
        threshold may need to be adjusted accordingly."""

    DEFAULT_COMPONENT_WEIGHTS = Components()
    """Default component weights. Actual default values are specified in :class:`Components`
    to allow adding new components without breaking existing usage."""

    LUMA_ONLY_WEIGHTS = Components(
        delta_hue=0.0,
        delta_sat=0.0,
        delta_lum=1.0,
        delta_edges=0.0,
    )
    """Component weights to use if `luma_only` is set."""

    FRAME_SCORE_KEY = "content_val"
    """Key in statsfile representing the final frame score after weighed by specified components."""

    METRIC_KEYS = [FRAME_SCORE_KEY, *Components._fields]
    """All statsfile keys this detector produces."""

    @dataclass
    class _FrameData:
        """Data calculated for a given frame."""

        hue: numpy.ndarray
        """Frame hue map [2D 8-bit]."""
        sat: numpy.ndarray
        """Frame saturation map [2D 8-bit]."""
        lum: numpy.ndarray
        """Frame luma/brightness map [2D 8-bit]."""
        edges: ty.Optional[numpy.ndarray]
        """Frame edge map [2D 8-bit, edges are 255, non edges 0]. Affected by `kernel_size`."""

    def __init__(
        self,
        threshold: float = 27.0,
        threshold_soft: float = 200,
        min_scene_len: int = 15,
        min_softcut_len: int = 5, #软切过程最小帧数
        max_softcut_len: int = 48, #软切过程最大帧数
        hardcut_hist_diff: float = 0.4, #硬切过程色调直方图差异阈值
        softcut_hist_diff: float = 0.55, #软切过程色调直方图差异阈值
        weights: "ContentDetector.Components" = DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: ty.Optional[int] = None,
        filter_mode: FlashFilter.Mode = FlashFilter.Mode.MERGE,
    ):
        """
        Arguments:
            threshold: Threshold the average change in pixel intensity must exceed to trigger a cut.
            min_scene_len: Once a cut is detected, this many frames must pass before a new one can
                be added to the scene list. Can be an int or FrameTimecode type.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel for expanding detected edges. Must be odd integer
                greater than or equal to 3. If None, automatically set using video resolution.
            filter_mode: Mode to use when filtering cuts to meet `min_scene_len`.
        """
        super().__init__()
        self._threshold: float = threshold
        self._threshold_softcut: float = threshold_soft
        self._min_softcut_len: int = min_softcut_len
        self._min_scene_len: int = min_scene_len
        self._max_softcut_len: int = max_softcut_len
        self._hardcut_hist_diff: float = hardcut_hist_diff
        self._softcut_hist_diff: float = softcut_hist_diff
        self._last_above_threshold: ty.Optional[int] = None
        self._last_frames: ty.List[ContentDetector._FrameData] = []
        self._weights: ContentDetector.Components = weights
        if luma_only:
            self._weights = ContentDetector.LUMA_ONLY_WEIGHTS
        self._kernel: ty.Optional[numpy.ndarray] = None
        if kernel_size is not None:
            if kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd integer >= 3")
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        self._frame_score: ty.Optional[float] = None
        self._frame_score_soft: ty.Optional[float] = None
        # TODO(v0.7): Handle timecodes in filter.
        self._flash_filter = FlashFilter(mode=filter_mode, length=min_scene_len)

    def get_metrics(self):
        return ContentDetector.METRIC_KEYS
    
    '''
    一个用来评估以这里为分界点，左边数字小，右边数字大，是否合理的评分
    '''
    def split_score(self, arr, idx, lambda_var=0.1, lambda_smooth = 5):
        left = arr[:idx]
        right = arr[idx:]
        mean_left = np.mean(left)
        mean_right = np.mean(right)
        var_left = np.var(left)
        var_right = np.var(right)
        right_diff = np.abs(np.diff(right)).mean() #惩罚右边不平滑
        smoothness_right = lambda_smooth / (right_diff + 0.5)
        score = (mean_right - mean_left) * (len(arr) - idx + 1) - lambda_var * (var_left + var_right) + smoothness_right
        return score

    """
        我想修改这部分的代码，使它可以识别“软切”
        不再记录之前一帧，而是记录之前若干帧
        首先判断当前帧和前一帧的差异，如果发现差异过大，则认为发生了“硬切”，直接切，并且清空之前若干帧的序列
        否则，比较这一帧和序列首部的帧的差异，如果差异过大，则认为发生了“软切”
        但这样会不会造成误判？它真的是“软切”而不是一个镜头内的场景变化吗？
        尝试引入“线性变化”的判断，只考虑普通的交叉溶解的话，颜色直方图的变化应该是线性的。
        此外，引入“双峰”判断，考虑如果是一个镜头，颜色直方图应该呈现平缓移动，而“软切”会呈现出“双峰”的特征。
    """
    def _calculate_frame_score(self, timecode: FrameTimecode, frame_img: numpy.ndarray) -> float:
        """Calculate score representing relative amount of motion in `frame_img` compared to
        the last time the function was called (returns 0.0 on the first call)."""
        # TODO: Add option to enable motion estimation before calculating score components.
        # TODO: Investigate methods of performing cheaper alternatives, e.g. shifting or resizing
        # the frame to simulate camera movement, using optical flow, etc...

        # Convert image into HSV colorspace.
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
        # print(hue.shape)

        # Performance: Only calculate edges if we have to.
        calculate_edges: bool = (self._weights.delta_edges > 0.0) or self.stats_manager is not None
        edges = self._detect_edges(lum) if calculate_edges else None

        if self._last_frames == []:
            # Need another frame to compare with for score calculation.
            self._last_frames.append(ContentDetector._FrameData(hue, sat, lum, edges))
            return 0.0

        score_components = ContentDetector.Components(
            delta_hue=_mean_pixel_distance(hue, self._last_frames[-1].hue),
            delta_sat=_mean_pixel_distance(sat, self._last_frames[-1].sat),
            delta_lum=_mean_pixel_distance(lum, self._last_frames[-1].lum),
            delta_edges=(
                0.0 if edges is None else _mean_pixel_distance(edges, self._last_frames[-1].edges)
            ),
        )

        frame_score: float = sum(
            component * weight for (component, weight) in zip(score_components, self._weights)
        ) / sum(abs(weight) for weight in self._weights)

        # Record components and frame score if needed for analysis.
        if self.stats_manager is not None:
            metrics = {self.FRAME_SCORE_KEY: frame_score}
            metrics.update(score_components._asdict())
            self.stats_manager.set_metrics(timecode, metrics)

        # 计算色调直方图差异，看是否有“双峰”
        hist1 = cv2.calcHist([hue], [0], None, [180], [0, 180])
        hist2 = cv2.calcHist([self._last_frames[-1].hue], [0], None, [180], [0, 180])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        frame_score *= min(diff, self._hardcut_hist_diff) / self._hardcut_hist_diff

        logging.info(f"hardcut {timecode} {diff}")

        # Store all data required to calculate the next frame's score.
        self._last_frames.append(ContentDetector._FrameData(hue, sat, lum, edges))

        return frame_score
    
    def _calculate_frame_score_soft(self, timecode: FrameTimecode, frame_img: numpy.ndarray) -> float:
        # 原代码的逻辑很奇怪，把计算和append写到了一起，所以这里就默认序列最后一个是当前图像了
        if len(self._last_frames) <= self._min_softcut_len + self._min_scene_len:
            return 0.0, 0.0
        hues = [frame.hue for frame in self._last_frames[max(0, len(self._last_frames) - (self._max_softcut_len + self._min_scene_len)):]]
        sats = [frame.sat for frame in self._last_frames[max(0, len(self._last_frames) - (self._max_softcut_len + self._min_scene_len)):]]
        lums = [frame.lum for frame in self._last_frames[max(0, len(self._last_frames) - (self._max_softcut_len + self._min_scene_len)):]]
        edges = [frame.edges for frame in self._last_frames[max(0, len(self._last_frames) - (self._max_softcut_len + self._min_scene_len)):]]
        hues_cnt = []
        sats_cnt = []
        lums_cnt = []
        edges_cnt = []
        for i in range(1, len(hues)):
            hues_cnt.append(_mean_pixel_distance(hues[i], hues[i - 1]))
            sats_cnt.append(_mean_pixel_distance(sats[i], sats[i - 1]))
            lums_cnt.append(_mean_pixel_distance(lums[i], lums[i - 1]))
            edges_cnt.append(0.0 if edges[i] is None else _mean_pixel_distance(edges[i], edges[i - 1]))
        mxscore = 0
        mxscore_i = -1
        for i in range(self._min_scene_len, len(hues_cnt) - self._min_softcut_len):
            score = self.split_score(hues_cnt, i) * self._weights.delta_hue + \
                    self.split_score(sats_cnt, i) * self._weights.delta_sat + \
                    self.split_score(lums_cnt, i) * self._weights.delta_lum + \
                    self.split_score(edges_cnt, i) * self._weights.delta_edges
            # print('score:', score)
            if score > mxscore:
                mxscore = score
                mxscore_i = i
        if mxscore_i == -1:
            return 0.0, 0.0
        
        # 计算当前帧（认为是软切之后的）和剪切点之前的帧的差别，是否超过了阈值
        score_components = ContentDetector.Components(
            delta_hue=_mean_pixel_distance(hues[max(0, mxscore_i - 3)], hues[-1]),
            delta_sat=_mean_pixel_distance(sats[max(0, mxscore_i - 3)], sats[-1]),
            delta_lum=_mean_pixel_distance(lums[max(0, mxscore_i - 3)], lums[-1]),
            delta_edges=(
                0.0 if edges[max(0, mxscore_i)] is None else _mean_pixel_distance(edges[max(0, mxscore_i)], self._last_frames[-1].edges)
            ),
        )

        frame_score: float = sum(
            component * weight for (component, weight) in zip(score_components, self._weights)
        ) / sum(abs(weight) for weight in self._weights)

        if frame_score < self._threshold:
            return 0.0, frame_score
        
        # 计算色调直方图差异，看是否有“双峰”
        hist1 = cv2.calcHist([hues[max(0, mxscore_i - 3)]], [0], None, [180], [0, 180])
        hist2 = cv2.calcHist([hues[-1]], [0], None, [180], [0, 180])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        logging.info(f"softcut {timecode} {diff}")

        if diff < self._softcut_hist_diff:
            return 0.0, frame_score
        return mxscore, frame_score

    def process_frame(
        self, timecode: FrameTimecode, frame_img: numpy.ndarray
    ) -> ty.List[FrameTimecode]:
        """Process the next frame. `frame_num` is assumed to be sequential.

        Args:
            frame_num (int): Frame number of frame that is being passed. Can start from any value
                but must remain sequential.
            frame_img (numpy.ndarray or None): Video frame corresponding to `frame_img`.

        Returns:
           ty.List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        self._frame_score = self._calculate_frame_score(timecode, frame_img)

        if self._frame_score is None:
            return []
        
        self._frame_score_soft, diff1 = self._calculate_frame_score_soft(timecode, frame_img)

        above_threshold: bool = self._frame_score >= self._threshold or (self._frame_score_soft is not None and self._frame_score_soft >= self._threshold_softcut)
        logging.info(f"{timecode} {self._frame_score} {self._frame_score_soft} {diff1} {self._threshold} {above_threshold}")
        ret = self._flash_filter.filter(timecode=timecode, above_threshold=above_threshold)
        if ret: #非空，即确实是剪切点
            self._last_frames = self._last_frames[-1:] #只保留最后一帧，也就是新镜头的第一帧
        return ret


    def _detect_edges(self, lum: numpy.ndarray) -> numpy.ndarray:
        """Detect edges using the luma channel of a frame.

        Arguments:
            lum: 2D 8-bit image representing the luma channel of a frame.

        Returns:
            2D 8-bit image of the same size as the input, where pixels with values of 255
            represent edges, and all other pixels are 0.
        """
        # Initialize kernel.
        if self._kernel is None:
            kernel_size = _estimated_kernel_size(lum.shape[1], lum.shape[0])
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)

        # Estimate levels for thresholding.
        # TODO: Add config file entries for sigma, aperture/kernel size, etc.
        sigma: float = 1.0 / 3.0
        median = numpy.median(lum)
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))

        # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
        # This increases edge overlap leading to improved robustness against noise and slow
        # camera movement. Note that very large kernel sizes can negatively affect accuracy.
        edges = cv2.Canny(lum, low, high)
        return cv2.dilate(edges, self._kernel)

    @property
    def event_buffer_length(self) -> int:
        return self._flash_filter.max_behind

'''
实际上，对于软切的部分，返回的值已经很正确了，基本上就是出现软切，返回一个大几百的值，否则一般都是0
但是好像原本判断硬切的部分就不太准。    需要调参，并且追加了“双峰”条件，对运动中的镜头的判断更准确了，即不会把它判断成硬切或软切，测试比原来的效果好
而且不知道为什么阈值改不了？    改了，要在config.py里面改
'''