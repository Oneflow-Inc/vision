from numpy import single
import oss2
import os
import json


class _OSS(object):
    _OSS_END_POINT = 'https://oss-cn-beijing.aliyuncs.com'
    _OSS_BUCKET_NAME = 'oneflow-ci-cache'
    _OSS_ACCESS_KEY_ID = os.environ['OSS_ACCESS_KEY_ID']
    _OSS_ACCESS_KEY_SECRET = os.environ['OSS_ACCESS_KEY_SECRET']

    def __init__(self) -> None:
        auth = oss2.Auth(self._OSS_ACCESS_KEY_ID, self._OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(
            auth, self._OSS_END_POINT, self._OSS_BUCKET_NAME)
        self.remote_dir = 'benchmark/'
        self.local_dir = '.benchmark-cache'

    def set_remote_dir(self, dir):
        self.remote_dir = dir

    def set_local_dir(self, dir):
        self.local_dir = dir

    def pull(self, remote_name, local_name):
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        remote_file = self.remote_dir + remote_name
        local_file = self.local_dir + local_name
        if self.bucket.object_exists(remote_file):
            return self.bucket.get_object_to_file(remote_file, local_file)
        else:
            return False

    def push(self, remote_name, local_name):
        remote_file = self.remote_dir + remote_name
        local_file = self.local_dir + local_name
        self.bucket.put_object_from_file(remote_file, local_file)


oss_instance = _OSS()
