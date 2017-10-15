from random import randint

# siganture matrix is partitioned into b bands consisting of r rows each
b = 50 # the more bands the less false negatives
r = 25 # the more rows the less false positives
# -> chosen by cross-validation

# size of one signature = number of min-hashing functions used
min_hash_size = r*b

# large prime numbers used by the hash functions
prime1 = 1400305337
prime2 = 1400305369

# size of output spaces of the hash functions
shingle_space_size = 8193
band_hash_bucket_size = 10000000 # as large as we can afford

# u.a.r integers a and b used for min-hashing/permutations
# a and b consist of min_hash_size elements
a = [randint(1, shingle_space_size) for i in range(min_hash_size)]
b = [randint(0, shingle_space_size) for i in range(min_hash_size)]

# u.a.r integers a and b used for hashing bands
# c and d consist of r elements
c = [randint(1, band_hash_bucket_size) for i in range(r)]
d = [randint(0, band_hash_bucket_size) for i in range(r)]

# calculate the Jaccard similarity between two lists
def similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2) * 1.0/len(set1 | set2)


def mapper(key, value):
    # key: None
    # value: one line of input file i.e a page

    # extract shingles from this page
    shingles = map(int, value.split()[1:])

    # compute signature of this page
    signature = []
    for i in range(min_hash_size):
        temp = []
        # go over all shingles and compute permuted index with a hash function
        for z in shingles:
            temp.append(((a[i]*z + b[i]) % prime1) % shingle_space_size)
        # take the minimal permuted index
        signature.append(min(temp))

    # split the signature into b bands each consisting of r rows
    bands = [signature[i:i + r] for i in range(0, len(signature), r)]

    # hash the bands of the signature (AND construction)
    band_hashes = []
    for band in bands:
        temp = []
        # hash a single band by summing over r hash functions
        for j in range(r):
            temp.append(((c[j]*band[j] + d[j]) % prime2) % band_hash_bucket_size)
        band_hashes.append(sum(temp) % band_hash_bucket_size)


    # key: a band's id concatenated with the band's hash
    # value: the page (input value)
    for i in range(len(band_hashes)):
        key = str(i) + "," + str(band_hashes[i])
        yield key, value


def reducer(key, values):
    # key: a band's id concatenated with the band's hash
    # values: list of pages having the same band hashed to the same bucket

    # sort the list of pages
    values.sort()

    # extract id and shingles from each page s.t. jaccard similarity can be
    # computed between candidate pairs
    page_ids = []
    page_shingles = []
    for value in values:
        # extract the page id
        page_id = int(value.split()[0].split("_")[1])
        # extract the shingles of a page
        shingles = map(int, value.split()[1:])
        page_ids.append(page_id)
        page_shingles.append(shingles)

    # for all candidate pairs (i,j) compute jaccard similarity
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            # for each candidate pair of pages calculate the similarity
            # if at least 0.85, output the page id's of the pair
            if similarity(page_shingles[i], page_shingles[j]) >= 0.85:
                yield page_ids[i], page_ids[j]
