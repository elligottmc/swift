
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

        #If you don't change SHARD_CONTAINER_SIZE to something very small, you will be waiting a long time
        threshold = ContainerSharder(self.conf).shard_container_size
        object_count = int(client.get_container(self.url, self.token, self.container)[0]['x-container-object-count'])

        if object_count > 0:
            raise Exception("Container is not empty.  Create fresh container or choose an empty one")

        else:

            new_object_count = 0
            for obj in xrange(threshold + 1):
                obj_file = open(str(obj), 'a+')
                obj_file.close()
                client.put_object(self.url, self.token, self.container, \
                                  obj_file.name, obj_file.name)
                os.remove(obj_file.name)
                new_object_count += 1

            #This can take a while if you haven't dialed down the sharding audit interval
            time.sleep(int(self.conf['container-sharder']['interval']))

        return new_object_count

    @staticmethod
    def find_paths(pattern, root_path):

        '''Simple method to find paths
        An external function and decorator seemed unnecessary'''

        found = []

        for path, _, files in os.walk(os.path.abspath(root_path)):
            for filename in fnmatch.filter(files, pattern):
                if filename not in found:
                    found.append(filename)
                    yield os.path.join(path, filename)


    def query_containers(self):

        '''Find shards for the given root container and
        retrieve metadata and object counts'''

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
                        objlist = []
                        for obj in objects:
                            objlist.append(obj[0])
                        dbdict['objects'] = objlist
                        dblist.append(dbdict)

        return dblist


    def find_shards(self):

        '''The time required to shard can be unpredictable.
        Very broadly, as the number of containers/shards grow, so does the time
        it takes to audit and shard all containers/shards
        Here we keep querying until shards are found '''

        query = None

        while True:
            query = self.query_containers()
            if len(query) > 0:
                break

        return self.query_containers()


class TestSharding(unittest.TestCase):

    '''Test the sharding functionality of sharding_split.
    Adjust params as appropriate.
    '''

    def setUp(self):
        self.start = time.time()
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

        '''Verify that a container pivots when the threshold is breached,
        that 2 and only 2 shards are created, that the number of objects in each shard
        is equivalent to the number of objects uploaded to the root container,
        and that each container db has appropriate metadata'''

        self.assertEqual(len(self.shards), 2)
        self.assertEqual(len(self.shards[0]['objects']) + \
                         len(self.shards[1]['objects']), self.obj_count)

        for obj in self.shards:
            for metastring in ("X-Container-Sysmeta-Shard-Container","X-Container-Sysmeta-Shard-Account"):
                self.assertIn(metastring, obj['meta'])

        end = time.time()
        print("Time from start to finish: %f" % (end - self.start))


    def tearDown(self):

        #Once it's possible to delete containers we should do that somewhere
        pass

if __name__ == '__main__':
    unittest.main()


