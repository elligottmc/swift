
'''Includes a class to experiment with sharding and test cases to verify
pivot sharding functions as per sharding_split spec'''

import os, fnmatch, time, unittest, sqlite3, random, string
import swiftclient.client as client
from swiftclient import ClientException
from swift.common.utils import readconf
from swift.container.sharder import ContainerSharder


class SampleSharding(object):

    '''Convenience class to play with pivot sharding for testing and experimentation.
    :param authurl: same as swiftclient authurl (type string)
    :param user:  same as swiftclient user (type string)
    :param key:  same as swiftclient key (type string)
    :param conf:  container-server config file with sharding parameters (type string)
    :param container: fresh or empty container to use for sample sharding (type string)
    '''

    def __init__(self, authurl, user, key, conf, container):

        auth = client.get_auth(authurl, user, key)
        self.url = auth[0]
        self.token = auth[1]
        self.conf = readconf(conf)
        self.container = container

    def enable_sharding(self):

        '''Enable sharding on the container
        with POST or PUT as appropriate'''

        try:
            client.get_container(self.url, self.token, self.container)

        except ClientException:
            client.put_container(self.url, self.token, self.container, \
                                 {'X-Container-Sharding': 'On'})
        else:
            client.post_container(self.url, self.token, self.container, \
                                  {'X-Container-Sharding': 'On'})


    def query_meta(self):

        '''Retrieve sharding header for verification'''

        try:
            get = client.get_container(self.url, self.token, self.container)

        except ClientException:
            print("Does container exist?")

        else:
            return bool(get[0]['x-container-sharding'])



    def create_sharding_condition(self):

        '''Determine sharding threshold from conf and saturate the container with
        enough objects to trigger the container sharder'''

        #If you don't change SHARD_CONTAINER_SIZE to something very small,
        #you will be waiting a long time
        threshold = ContainerSharder(self.conf).shard_container_size
        object_count = int(client.get_container(self.url, self.token, self.container)\
                               [0]['x-container-object-count'])

        if object_count > 0:
            raise Exception("Container is not empty.  \
            Create fresh container or choose an empty one")

        else:

            new_object_count = 0
            for obj in xrange(threshold + 1):
                obj_file = open(str(obj), 'a+')
                obj_file.close()
                client.put_object(self.url, self.token, self.container, \
                                  obj_file.name, obj_file.name)
                os.remove(obj_file.name)
                new_object_count += 1

        return new_object_count


    @staticmethod
    def find_paths(pattern, root_path):

        '''Generic method to find paths'''

        found = []

        for path, _, files in os.walk(os.path.abspath(root_path)):
            for filename in fnmatch.filter(files, pattern):
                if filename not in found:
                    found.append(filename)
                    yield os.path.join(path, filename)


    def query_containers(self):

        '''Find shards for the given root container
        and retrieve metadata and object counts'''

        dblist = []

        container_dbs = [db for db in self.find_paths('*.db', '/') if 'container' in db]

        for container_db in container_dbs:

            conn = sqlite3.connect(container_db)

            with conn:

                cur = conn.cursor()
                cur.execute("SELECT container, metadata FROM container_info")

                rows = cur.fetchall()

                for row in rows:
                    if self.container in row[0] and "X-Container-Sysmeta-Shard-Container" in row[1]:
                        dbdict = dict(meta=row[1])
                        objcur = conn.cursor()
                        objcur.execute("SELECT name from object")
                        objects = objcur.fetchall()
                        dbdict['objects'] = [obj[0] for obj in objects]
                        dblist.append(dbdict)

        return dblist


    def find_shards(self):

        '''The time required to shard can be unpredictable.
        Very broadly, as the number of containers/shards grow, so does the time
        it takes to audit and shard all containers/shards
        Here we keep querying until shards are found '''

        start = time.time()
        query = None

        while True:
            query = self.query_containers()
            if len(query) > 0:
                end = time.time()
                break

        print("Estimated Time to Shard: %f" % (end - start))
        return query


class TestSharding(unittest.TestCase):

    '''Test the sharding functionality of sharding_split.
    Adjust params as appropriate.
    '''

    def setUp(self):

        testauthurl = 'http://127.0.0.1:8080/auth/v1.0/'
        testuser = 'test:tester'
        testkey = 'testing'
        testconf = '/etc/swift/container-server/1.conf'
        testcontainer = ''.join(random.choice(string.ascii_lowercase + string.digits) \
                               for _ in range(15))
        shardme = SampleSharding(testauthurl, testuser, testkey, testconf, testcontainer)
        shardme.enable_sharding()
        self.obj_count = shardme.create_sharding_condition()
        self.meta = shardme.query_meta()
        self.shards = shardme.find_shards()

    def test_enable_sharding(self):

        '''Verify that X-Container-Sharding = True
        in client headers'''

        self.assertTrue(self.meta)


    def test_sharding(self):

        '''Verify that a container has sharded
        and that 2 and only 2 shards are created.'''

        self.assertEqual(len(self.shards), 2)


    def test_container_meta(self):

        '''Verify that each container db has appropriate metadata'''

        for obj in self.shards:
            for metastring in ("X-Container-Sysmeta-Shard-Container", \
                               "X-Container-Sysmeta-Shard-Account"):
                self.assertIn(metastring, obj['meta'])


if __name__ == '__main__':
    unittest.main()


