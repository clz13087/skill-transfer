import logging
from UDPmanager.UDP_client import UDP_Client
from UDPmanager.UDP_server import UDP_Server

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, lstmClientAddress, lstmClientPort, lstmServerAddress, lstmServerPort) -> None:
        self.lstmClientAddress = lstmClientAddress
        self.lstmClientPort = lstmClientPort
        self.lstmServerAddress = lstmServerAddress
        self.lstmServerPort = lstmServerPort

        self.client = UDP_Client()
        self.server = UDP_Server(ip=self.lstmServerAddress, port=self.lstmServerPort)
        self.server.receive_start()
        logger.info("set up lstm client and server")

    def predict_position_rotation(self, pos_rot_list):
        try:
            self.client.send(pos_rot_list, self.lstmClientAddress, self.lstmClientPort)

        except:
            logger.info("Can NOT send the pos for lstm server!")
            pass
        
        return self.server.data

        ###　一旦実装のしやすさも加味し，送るだけ送る，受け取った最新のデータを予測結果として扱うという手法を取る．
        ###　将来的には，送る時にloopcountの値も合わせて送る→予測し、loopcountに予測フレーム数を足した値を付加して返信→予測データが欲しい場合はそのフレームのloopcountを入力してpopという扱いにしたい．
        ###　そうすることで，何フレーム遅れているかなども計算しやすくなる．