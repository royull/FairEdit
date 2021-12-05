#include "../Graph.hpp"
#include "HelpFuncs.hpp"

//#include <random>
//#include <ctime>
//#include <limits>
#include<cstdio>
#include<thread>

extern "C"
{
	//#include<sys/types.h>
	#include<sys/stat.h>
	#include<fcntl.h>
}

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE HelperFunctions
#include <boost/test/unit_test.hpp>

#define TEST_FILE_PREFIX "wALk_tESt_"
#define BUF_SIZE 128

using namespace std;

char const * const testFile = TEST_FILE_PREFIX "input_simple";

char * trim(char * from)
{
	while(*from <= ' ') ++from;
	
	char * to = from + strlen(from);
	
	if(to > from) --to;
	
	while(*to <= ' ') *(to--) = '\0';
	
	return from;
}

BOOST_AUTO_TEST_CASE(GRAPH_SIMPLE)
{
	{
		Graph const g = Graph::readGraph("1,2", 0);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "1 -> 2\n");

		char walk[10];
		FastPRNG prng;
		g.doWalk(1, 2, prng, walk);

		BOOST_CHECK(!memcmp(walk, "1,2", 4));
		BOOST_CHECK_EQUAL(g.getNumNodes(), 2);
	}

	{
		Graph const g = Graph::readGraph("1,2\n\n\n\n", 0);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "1 -> 2\n");

		char walk[10];
		FastPRNG prng;
		g.doWalk(1, 2, prng, walk);

		BOOST_CHECK(!memcmp(walk, "1,2", 4));
	}

	{
		Graph const g = Graph::readGraph("3,2\n0,1\n2,1\n1,3", 0);
		string const str = g.toString();
		
		BOOST_CHECK_EQUAL(str, "0 -> 1\n1 -> 3\n2 -> 1\n3 -> 2\n");

		char walk[20];
		FastPRNG prng;
		g.doWalk(0, 6, prng, walk);

		BOOST_CHECK(!memcmp(walk, "0,1,3,2,1,3", 12));
		BOOST_CHECK_EQUAL(g.getNumNodes(), 4);
		BOOST_CHECK_EQUAL(g.getNode(2, 0), 1);
		BOOST_CHECK_EQUAL(g.getNumTransitions(), 4);
		BOOST_CHECK_EQUAL(g.getNumTransitions(0), 1);
		BOOST_CHECK_EQUAL(g.getTransition(1), 3);
	}
}

BOOST_AUTO_TEST_CASE(GRAPH_UNDIRECT)
{
	{
		Graph const g = Graph::readGraph("3,2\n0,1\n2,1\n1,3", Graph::flag_undirected);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "0 -> 1\n1 -> 3\n1 -> 0\n1 -> 2\n2 -> 1\n2 -> 3\n3 -> 2\n3 -> 1\n");

		char walk[10];
		FastPRNG prng;
		unsigned hit1 = 0;
		unsigned hit3 = 0;
		
		for(unsigned i = 0; i < 1000; ++i)
		{
			g.doWalk(2, 2, prng, walk);

			if(!memcmp(walk, "2,1", 4)) ++hit1;
			else if(!memcmp(walk, "2,3", 4)) ++hit3;
			else BOOST_CHECK(false);
		}
		
		// Check whether node 1 is hit about 1 of 2 times
		BOOST_WARN(hit1 >= 450 && hit1 <= 550);
		BOOST_CHECK(hit1 + hit3 == 1000);
	}

	{
		Graph const g = Graph::readGraph("1,2\n3,2\n2,3", Graph::flag_undirected);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "1 -> 2\n2 -> 3\n2 -> 1\n3 -> 2\n");
	}
}

BOOST_AUTO_TEST_CASE(GRAPH_REFLECT)
{
	{
		Graph const g = Graph::readGraph("3,2\n0,1\n2,1\n1,3", Graph::flag_reflect);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "0 -> 1\n1 -> 3\n1 -> 0\n1 -> 2\n2 -> 1\n2 -> 3\n3 -> 2\n3 -> 1\n");
	}

	{
		Graph const g = Graph::readGraph("1,2\n3,2\n2,3", Graph::flag_reflect);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "1 -> 2\n2 -> 3\n2 -> 1\n3 -> 2\n");
	}
}

BOOST_AUTO_TEST_CASE(GRAPH_ADVANCED_PROBABILITY_CHECK)
{
	{
		Graph const g = Graph::readGraph("3,2\n1,0\n2,1\n0,2\n1,3", 0);
		string const str = g.toString();

		BOOST_CHECK_EQUAL(str, "0 -> 2\n1 -> 0\n1 -> 3\n2 -> 1\n3 -> 2\n");

		char walk[2000];
		FastPRNG prng;

		g.doWalk(1, 1000, prng, walk);

		unsigned counts[4];
		counts[0] = std::count(walk, walk + 2000, '0');
		counts[1] = std::count(walk, walk + 2000, '1');
		counts[2] = std::count(walk, walk + 2000, '2');
		counts[3] = std::count(walk, walk + 2000, '3');

		// 0: 1/6
		BOOST_WARN(counts[0] >= 150);
		BOOST_WARN(counts[0] <= 200);

		// 1: 1/3
		BOOST_WARN(counts[1] >= 300);
		BOOST_WARN(counts[1] <= 400);

		// 2: 1/3
		BOOST_WARN(counts[2] >= 300);
		BOOST_WARN(counts[2] <= 400);

		// 3: 1/6
		BOOST_WARN(counts[3] >= 150);
		BOOST_WARN(counts[3] <= 200);
	}
}




