import zmq
import threading
import time

# both need to be on the same port
# client send a request, and then wait for a reply. It cannot send a second request until it receives the reply to the first one.
# server must wait for a request, and then send a reply. It cannot send a reply unless it just received a request.

def rep_server(port=5555, num_requests=5):
    """reply server"""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    print(f"reply server on port: {port}")
    responses = []

    for i in range(num_requests):
        message = socket.recv_string()
        print(f"server received: {message}")

        response = f"i gotchu {message}"
        time.sleep(0.5)

        socket.send_string(response)
        responses.append((message, response))

    socket.close()
    context.term()
    return responses

def req_client(port=5555, requests=None):
    """request client"""

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")

    print(f"request client on port: {port}")
    results = []

    for request in requests:
        print(f"client sending: {request}")
        socket.send_string(request)

        response = socket.recv_string()
        print(f"client received: {response}")
        results.append((request, response))

    socket.close()
    context.term()


port = 5555
server_thread = threading.Thread(target=rep_server, args=(port, 3))
server_thread.daemon = True
server_thread.start()

time.sleep(1)

client_results = req_client(port, ["what is prefill", "what is decode"])