import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from collections import defaultdict, deque

class MotionFilter:
    def __init__(self, buffer_size=30, cutoff=0.1, fs=30.0):
        self.buffer_size = buffer_size
        self.cutoff = cutoff
        self.fs = fs
        self.b, self.a = butter(N=2, Wn=cutoff, fs=fs, btype='low')
        # 各参加者のバッファを管理する辞書
        self.position_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.rotation_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))

    def apply_butterworth_filter(self, data_dict, mode='position'):
        """
        全参加者のデータに対してButterworthフィルターを適用
        :param data_dict: 参加者ごとのデータ辞書 (例: {'participant1': data1, ...})
        :param mode: 'position' または 'rotation'
        :return: フィルタ済みデータの辞書
        """
        filtered_data = {}
        # 適切なバッファを選択
        buffer = (
            self.position_buffers if mode == 'position' else self.rotation_buffers
        )

        for participant_key, data in data_dict.items():
            # バッファに新しいデータを追加
            buffer[participant_key].append(data)

            # バッファが十分でない場合は入力データをそのまま返す
            if len(buffer[participant_key]) < self.buffer_size:
                filtered_data[participant_key] = data
                continue

            # バッファ内のデータを配列に変換
            buffer_array = np.array(buffer[participant_key])

            # 各次元に対してフィルタを適用
            filtered_result = np.zeros_like(buffer_array)
            for i in range(buffer_array.shape[1]):
                filtered_result[:, i] = filtfilt(self.b, self.a, buffer_array[:, i])

            # 最新のフィルタ済みデータを保存
            filtered_data[participant_key] = filtered_result[-1]

        return filtered_data

    def InitLowPassFilter(self, samplerate, fp, fs, gpass: int = 3, gstop: int = 40):
        """
        Initialize low pass filter
        Order and Butterworth normalized frequency will calculate automatically.

        Parameters
        -----
        samplerate: int
            Sampling rate of filter
        fp: int
            Passband edge frequency
        fs: int
            Stopband edge frequency
        gpass: (Optional) int
            Passband edge maximum loss [db]
        gstop: (Optional) int
            Stopband edge minimum loss [db]

        """

        fn      = samplerate / 2     # Nyquist frequency
        wp      = fp / fn            # Normalize passband edge frequency with Nyquist frequency
        ws      = fs / fn            # Normalize stopband edge frequency with Nyquist frequency
        n, Wn   = signal.buttord(wp, ws, gpass, gstop)  # Calculate the order and Butterworth normalized frequency

        self.lowB, self.lowA = signal.butter(n, Wn, "low")
        self.lowB = self.lowB.tolist()
        self.lowA = self.lowA.tolist()

    def InitLowPassFilterWithOrder(self, samplerate, fp, n):
        """
        Initialize low pass filter with order.

        Parameters
        ----------
        samplerate: int
            The sample rate.
        fp: int
            The passband edge frequency.
        n: int
            The order of the filter
        """

        fn = samplerate / 2     # Nyquist frequency
        wn = fp / fn            # The critical frequency or frequencies

        self.lowB, self.lowA = signal.butter(n, wn, "low")
        self.lowB = self.lowB.tolist()
        self.lowA = self.lowA.tolist()
        self.lowN = n

    def InitHighPassFilter(self, samplerate, fp, fs, gpass: int = 3, gstop: int = 40):
        """
        Initialize high pass filter.
        Order and Butterworth normalized frequency will calculate automatically.

        Parameters
        -----
        samplerate: int
            Sampling rate of filter
        fp: int
            Passband edge frequency
        fs: int
            Stopband edge frequency
        gpass: (Optional) int
            Passband edge maximum loss [db]
        gstop: (Optional) int
            Stopband edge minimum loss [db]

        """

        fn      = samplerate / 2     # Nyquist frequency
        wp      = fp / fn            # Normalize passband edge frequency with Nyquist frequency
        ws      = fs / fn            # Normalize stopband edge frequency with Nyquist frequency
        n, Wn   = signal.buttord(wp, ws, gpass, gstop)  # Calculate the order and Butterworth normalized frequency

        self.highB, self.highA = signal.butter(n, Wn, "high")
        self.highB = self.highB.tolist()
        self.highA = self.highA.tolist()

    def InitHighPassFilterWithOrder(self, samplerate, fp, n):
        """
        Initialize high pass filter with order.

        Parameters
        ----------
        samplerate: int
            The sample rate.
        fp: int
            The passband edge frequency.
        n: int
            The order of the filter
        """

        fn = samplerate / 2     # Nyquist frequency
        wn = fp / fn            # The critical frequency or frequencies

        self.lowB, self.lowA = signal.butter(n, wn, "high")
        self.lowB = self.lowB.tolist()
        self.lowA = self.lowA.tolist()
        self.highN = n


    def lowpass(self, x_box, x_filt_box):  # x = [1, (self.N + 1)]
        """
        Hagi comment: ローパスっぽいがなんでこのやり方なのかわからない
        """

        y1_all = 0
        y2_all = 0
        for i in range(0, self.lowN + 1):
            y1 = self.lowB[i] * x_box[self.lowN - i]
            y1_all += y1
        for i in range(1, self.lowN + 1):
            y2 = self.lowA[i] * x_filt_box[self.lowN - i]
            y2_all += y2

        y = y1_all - y2_all
        return y

    def lowpass2(self, x_box, x_filt_box):
        y1_all = [0, 0, 0, 0, 0, 0, 0, 0]
        y2_all = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, self.lowN + 1):
            y1 = list(map(lambda x: x * self.lowB[i], x_box[self.lowN - i]))
            y1_all = list(map(lambda x, y: x + y, y1_all, y1))
        for i in range(1, self.lowN + 1):
            y2 = list(map(lambda x: x * self.lowA[i], x_filt_box[self.lowN - i]))
            y2_all = list(map(lambda x, y: x + y, y2_all, y2))

        y = list(map(lambda x, y: x - y, y1_all, y2_all))
        return np.array(y)

    def ButterFilter(self, x):
        """
        Butterworth filter

        Parameters
        ----------
        x: array_like
            The array of data to be filtered.

        Returns
        ----------
        y: ndarray
            The filtered output with the same shape as x.
        """

        y = signal.filtfilt(self.lowB, self.lowA, x)
        return y

    def HighPassFilter(self, x):
        """
        High pass filter

        Parameters
        ----------
        x: array_like
            The array of data to be filtered.

        Returns
        ----------
        y: ndarray
            The filtered output with the same shape as x.
        """

        y = signal.filtfilt(self.highB, self.highA, x)
        return y
