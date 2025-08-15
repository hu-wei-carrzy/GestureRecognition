import cv2
import mediapipe as mp
import time
import handle

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# 手部关键点检测方法
# static_image_mode： 默认为 False，将输入图像视为视频流。
# max_num_hands： 最多检测几只手，默认为2
# min_detection_confidence： 手部检测模型的最小置信值（0-1之间），超过阈值则检测成功。默认为 0.5
# min_tracking_confidence： 坐标跟踪模型的最小置信值 (0-1之间)，用于将手部坐标视为成功跟踪，不成功则在下一个输入图像上自动调用手部检测
# 返回值
# MULTI_HAND_LANDMARKS： 被检测/跟踪的手的集合，其中每只手被表示为21个手部地标的列表，每个地标由x, y, z组成。x和y分别由图像的宽度和高度归一化为[0,1]。Z表示地标深度。
# MULTI_HANDEDNESS： 被检测/追踪的手是左手还是右手的集合。每只手由label(标签)和score(分数)组成。 label 是 ‘Left’ 或 ‘Right’ 值的字符串。 score 是预测左右手的估计概率。
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame= cv2.flip(frame,1)
    # process检测
    # 函数process接收RGB格式的numpy数组，返回包含3个字段的具名元组（NamedTuple）
    # multi_hand_landmarks：每只手的关节点坐标。
    # multi_hand_world_landmarks：每只手的关节点在真实世界中的3D坐标（以米m为单位），原点位于手的近似几何中心。
    # multi_handedness：每只手的手性（左 / 右手）
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 动作映射（所有手的关节点坐标）results.multi_hand_landmarks
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks_ in enumerate(results.multi_hand_landmarks):
            # hand_landmarks_ 动作映射（每只手的关节点坐标）
            handle.Handle(hand_landmarks_, mp_hands)

        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pass

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # 无目标 当前时间  有目标时时间 感觉判断无必要
        if time.time() - handle.handIntime > 0.1:
            handle.nowStart = ''
            # print("无目标")

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()