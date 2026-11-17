###
# Connection helper, same as the lab's kinova/manual/utilities.py (Kinova Kortex examples).
# KINOVA (R) KORTEX (TM) -- Copyright (c) 2018 Kinova inc. All rights reserved.
# This software may be modified and distributed under the terms of the BSD 3-Clause license.
###
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.autogen.messages import Session_pb2


class DeviceConnection:
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(ip, username, password):
        """Returns a context manager giving the RouterClient used to create services."""
        return DeviceConnection(ip, port=DeviceConnection.TCP_PORT, credentials=(username, password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials=("", "")):
        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials
        self.sessionManager = None
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ipAddress, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)
        return self.router

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sessionManager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.sessionManager.CloseSession(router_options)
        self.transport.disconnect()
