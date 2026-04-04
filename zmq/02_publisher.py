import zmq
import threading
import time
import random

# a messaging pattern where publishers send messages to a topic without knowing the consumers (subscribers)

def publisher(port=5556, num_messages=10):
    """pub publisher"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")

    print(f"publisher started on port: {port}")
    time.sleep(1)

    topics = ['news', 'weather', 'sports']
    messages_sent = []

    for i in range(num_messages):
        topic = random.choice(topics)
        message = f"{topic.upper()} message {i+1}: data {random.randint(1, 100)}"

        socket.send_string(f"{topic} {message}")
        print(f"publisher sent: [{topic}] {message}")
        messages_sent.append((topic, message))

        time.sleep(0.3)

    socket.close()
    context.term()
    return messages_sent

def subscriber(port=5556, topic_filter="", name="subscriber"):
    """subscriber"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")

    socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)

    print(f"{name} subscribed to topic: '{topic_filter if topic_filter else 'ALL'}'")
    messages_received = []

    for i in range(5):
        try:
            message = socket.recv_string(flags=zmq.NOBLOCK)
            print(f"{name} received: {message}")
            messages_received.append(message)
        except zmq.Again:
            pass
        time.sleep(0.1)

    socket.close()
    context.term()
    return messages_received


port = 5556
publisher_thread = threading.Thread(target=publisher, args=(port, 8))
publisher_thread.daemon = True
publisher_thread.start()

time.sleep(1.5)

subscribers = []
threads = []

# topic filter 
subscriber_configs = [
    ("news", "subscriber1"),
    ("weather", "subscriber2"),
    ("sports", "subscriber3")
]

for topic_filter, name in subscriber_configs:
    thread = threading.Thread(target=lambda tf=topic_filter, n=name: subscribers.append(subscriber(port, tf, n)))
    thread.daemon = True
    thread.start()
    threads.append(thread)
    time.sleep(0.2)