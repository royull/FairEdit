#include "../HelperFunctions.hpp"
#include <random>
#include <ctime>
#include <limits>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE HelperFunctions
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(BINARY_FIND)
{
	uint const arr[] = {1, 4, 9, 12, 13, 57, 98, 100, 235, 237};
	size_t const items = sizeof(arr)/sizeof(*arr);
	
	BOOST_CHECK_EQUAL(binaryFind(arr, arr + items, (uint)1), arr);
	BOOST_CHECK_EQUAL(binaryFind(arr, arr + items, (uint)237), arr + items - 1);
	BOOST_CHECK_EQUAL(binaryFind(arr, arr + items, (uint)13), arr + 4);
}

BOOST_AUTO_TEST_CASE(COUNT)
{
	char const * const text = "Das ist mein Text\n\n\x00\t Das ist super, oder?\r\n";
	
	BOOST_CHECK_EQUAL(count(text, text + 44, 'D'), 2);
	BOOST_CHECK_EQUAL(count(text, text + 44, '\r'), 1);
	BOOST_CHECK_EQUAL(count(text, text + 44, '\0'), 1);
	BOOST_CHECK_EQUAL(count(text, text + 44, '\n'), 3);
}

BOOST_AUTO_TEST_CASE(CONVERT_INT)
{
	BOOST_CHECK_EQUAL(toInt("10"), 10);
	BOOST_CHECK_EQUAL(toInt("9563429"), 9563429);

	char str[11];

	std::minstd_rand g(time(0));
	std::uniform_int_distribution<uint> distr;
	
	for(uint i = 0; i < 256; ++i)
	{
		uint const rnd = distr(g);
		unsigned const len = writeInt(str, rnd);
		str[len] = '\0';
		BOOST_CHECK_EQUAL(toInt(str), rnd);
	}
}

BOOST_AUTO_TEST_CASE(BOOL_SORT)
{
	uint list[] = {8, 29, 856, 36, 956, 34, 435, 64};
	uint mustBe[] = {8, 29, 34, 36, 64, 435, 856, 956};
	
	boolSort(list, 8);
	
	BOOST_CHECK(!memcmp(list, mustBe, sizeof(list)));
}

BOOST_AUTO_TEST_CASE(COUNT_SORT_2D_1)
{
	uint list[] = {8, 0, 7, 3, 9, 6, 0, 6, 3, 7, 8, 4, 13, 0, 8, 2};
	uint mustBe[] = {0, 6, 3, 7, 7, 3, 8, 0, 8, 4, 8, 2, 9, 6, 13, 0};
	
	countSort2D(list, 2, 8);
	
	BOOST_CHECK(!memcmp(list, mustBe, sizeof(list)));
}

BOOST_AUTO_TEST_CASE(COUNT_SORT)
{
	uint list[] = {8, 7, 9, 0, 3, 8, 13, 8};
	uint const mustBe[] = {0, 3, 7, 8, 8, 8, 9, 13};
	
	countSort(list, 8);
	
	BOOST_CHECK(!memcmp(list, mustBe, sizeof(list)));
}

BOOST_AUTO_TEST_CASE(COUNT_SORT_2D_2)
{
	uint list1[] = {8, 7, 9, 0, 3, 8, 13, 8};
	uint list2[] = {0, 3, 6, 6, 7, 4, 0, 2};
	
	uint const mustBe1[] = {0, 3, 7, 8, 8, 8, 9, 13};
	uint const mustBe2[] = {6, 7, 3, 0, 4, 2, 6, 0};
	
	countSort2D({list1, list2}, 8);
	
	BOOST_CHECK(!memcmp(list1, mustBe1, sizeof(list1)));
	BOOST_CHECK(!memcmp(list2, mustBe2, sizeof(list2)));
}

BOOST_AUTO_TEST_CASE(TO_INT)
{
	char const str1[] = "123";
	char const str2[] = "0";
	char const str3[] = "";
	char const str4[] = "4294967295";
	
	BOOST_CHECK_EQUAL(toInt(str1), 123);
	BOOST_CHECK_EQUAL(toInt(str2), 0);
	BOOST_CHECK_EQUAL(toInt(str3), 0);
	BOOST_CHECK_EQUAL(toInt(str4), 4294967295);
}

BOOST_AUTO_TEST_CASE(WRITE_INT)
{
	
	//TODO
}

BOOST_AUTO_TEST_CASE(WRITE_TO_INT)
{
	
}



