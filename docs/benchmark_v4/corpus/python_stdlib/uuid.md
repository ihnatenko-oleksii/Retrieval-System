# Python standard library: `uuid`

Official source: https://docs.python.org/3/library/uuid.html

### Command-Line Usage

Added in version 3.12.

The uuid module can be executed as a script from the command line.

python -m uuid [-h] [-u {uuid1,uuid3,uuid4,uuid5,uuid6,uuid7,uuid8}] [-n NAMESPACE] [-N NAME]

The following options are accepted:

-h, --help

Show the help message and exit.

-u <uuid>

--uuid <uuid>

Specify the function name to use to generate the uuid. By default uuid4() is used.

Changed in version 3.14: Allow generating UUID versions 6, 7 and 8.

-n <namespace>

--namespace <namespace>

The namespace is a UUID, or @ns where ns is a well-known predefined UUID addressed by namespace name. Such as @dns, @url, @oid, and @x500. Only required for uuid3() / uuid5() functions.

-N <name>

--name <name>

The name used as part of generating the uuid. Only required for uuid3() / uuid5() functions.

-C <num>

--count <num>

Generate num fresh UUIDs.

Added in version 3.14.

### Example

Here are some examples of typical usage of the uuid module:

>>> import uuid >>> # make a UUID based on the host ID and current time >>> uuid.uuid1() UUID('a8098c1a-f86e-11da-bd1a-00112444be1e') >>> # make a UUID using an MD5 hash of a namespace UUID and a name >>> uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org') UUID('6fa459ea-ee8a-3ca4-894e-db77e160355e') >>> # make a random UUID >>> uuid.uuid4() UUID('16fd2706-8baf-433b-82eb-8c7fada847da') >>> # make a UUID using a SHA-1 hash of a namespace UUID and a name >>> uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org') UUID('886313e1-3b8a-5372-9b90-0c9aee199e5d') >>> # make a UUID from a string of hex digits (braces and hyphens ignored) >>> x = uuid.UUID('{00010203-0405-0607-0809-0a0b0c0d0e0f}') >>> # convert a UUID to a string of hex digits in standard form >>> str(x) '00010203-0405-0607-0809-0a0b0c0d0e0f' >>> # get the raw 16 bytes of the UUID >>> x.bytes b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f' >>> # make a UUID from a 16-byte string >>> uuid.UUID(bytes=x.bytes) UUID('00010203-0405-0607-0809-0a0b0c0d0e0f') >>> # get the Nil UUID >>> uuid.NIL UUID('00000000-0000-0000-0000-000000000000') >>> # get the Max UUID >>> uuid.MAX UUID('ffffffff-ffff-ffff-ffff-ffffffffffff') >>> # same as UUIDv1 but with fields reordered to improve DB locality >>> uuid.uuid6() UUID('1f0799c0-98b9-62db-92c6-a0d365b91053') >>> # get UUIDv7 creation (local) time as a timestamp in milliseconds >>> u = uuid.uuid7() >>> u.time 1743936859822 >>> # get UUIDv7 creation (local) time as a datetime object >>> import datetime as dt >>> dt.datetime.fromtimestamp(u.time / 1000) datetime.datetime(...) >>> # make a UUID with custom blocks >>> uuid.uuid8(0x12345678, 0x9abcdef0, 0x11223344) UUID('00001234-5678-8ef0-8000-000011223344')

### Command-Line Example

Here are some examples of typical usage of the uuid command-line interface:

# generate a random UUID - by default uuid4() is used $ python -m uuid # generate a UUID using uuid1() $ python -m uuid -u uuid1 # generate a UUID using uuid5 $ python -m uuid -u uuid5 -n @url -N example.com # generate 42 random UUIDs $ python -m uuid -C 42
