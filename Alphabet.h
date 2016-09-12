#ifndef _ALPHABET_
#define _ALPHABET_

#include "MyLib.h"

/*
 please check to ensure that m_size not exceeds the upbound of int
 */

/*
	This class serializes feature from string to int.
	Index starts from 0.
	*/

/**
 * The basic class of quark class.
 *  @param  std::string        String class name to be used.
 *  @param  int         ID class name to be used.
 *  @author Naoaki Okazaki
 */
class basic_quark {
	static const  int max_freq = 100000;
	static const  int max_capacity = 10000000;
protected:
	typedef unordered_map<std::string, int> StringToId;
	typedef std::vector<std::string> IdToString;

	StringToId m_string_to_id;
	IdToString m_id_to_string;
	std::vector<int> m_freqs;
	bool m_b_fixed;
	int m_size;

public:
	/**
	 * Construct.
	 */
	basic_quark()
	{
		clear();

	}

	/**
	 * Destruct.
	 */
	virtual ~basic_quark()
	{
	}

	/**
	 * Map a string to its associated ID.
	 *  If string-to-integer association does not exist, allocate a new ID.
	 *  @param  str         String value.
	 *  @return           Associated ID for the string value.
	 */
	int operator[](const std::string& str)
	{
		StringToId::const_iterator it = m_string_to_id.find(str);
		if (it != m_string_to_id.end()) {
			return it->second;
		}
		else if (!m_b_fixed){
			int newid = m_size;
			if (newid > 0 && m_freqs[newid - 1] != max_freq){
				std::cout << "not a descended sorted alphabet!" << std::endl;
				assert(0);
			}
			m_freqs.push_back(max_freq);
			m_id_to_string.push_back(str);
			m_string_to_id.insert(std::pair<std::string, int>(str, newid));
			m_size++;
			if (m_size >= max_capacity)m_b_fixed = true;
			return newid;
		}
		else
		{
			return -1;
		}
	}


	/**
	 * Convert ID value into the associated string value.
	 *  @param  qid         ID.
	 *  @param  def         Default value if the ID was out of range.
	 *  @return           String value associated with the ID.
	 */
	const std::string& from_id(const int& qid, const std::string& def = "") const
	{
		if (qid < 0 || m_size <= qid) {
			return def;
		}
		else {
			return m_id_to_string[qid];
		}
	}



	/**
	 * Convert string value into the associated ID value.
	 *  @param  str         String value.
	 *  @return           ID if any, otherwise -1.
	 */
	int from_string(const std::string& str, int curfreq = max_freq)
	{
		StringToId::const_iterator it = m_string_to_id.find(str);
		if (it != m_string_to_id.end()) {
			return it->second;
		}
		else if (!m_b_fixed){
			int normfreq = curfreq < max_freq ? curfreq : max_freq;
			int newid = m_size;
			if (newid > 0 && m_freqs[newid - 1] < normfreq){
				std::cout << "not a descended sorted alphabet!" << std::endl;
				assert(0);
			}
			m_freqs.push_back(normfreq);
			m_id_to_string.push_back(str);
			m_string_to_id.insert(std::pair<std::string, int>(str, newid));
			m_size++;
			if (m_size >= max_capacity)m_b_fixed = true;
			return newid;
		}
		else
		{
			return -1;
		}
	}

	void clear()
	{
		m_string_to_id.clear();
		m_id_to_string.clear();
		m_freqs.clear();
		m_b_fixed = false;
		m_size = 0;
	}

	void set_fixed_flag(bool bfixed)
	{
		m_b_fixed = bfixed;
	}

	/**
	 * Get the number of string-to-id associations.
	 *  @return           The number of association.
	 */
	size_t size() const
	{
		return m_size;
	}


	void read(std::ifstream &inf)
	{
		clear();
		static string tmp;
		my_getline(inf, tmp);
		chomp(tmp);
		m_size = atoi(tmp.c_str());
		assert(m_size >= 0);
		std::vector<std::string> featids;
		for (int i = 0; i < m_size; ++i) {
			my_getline(inf, tmp);
			split_bychars(tmp, featids);
			m_string_to_id[featids[0]] = i;
			assert(atoi(featids[1].c_str()) == i);
			if (featids.size() > 2){
				m_freqs[i] = atoi(featids[2].c_str());
			}
		}
	}

	void write(std::ofstream &outf) const
	{
		outf << m_size << std::endl;
		for (int i = 0; i < m_size; i++)
		{
			if (m_freqs[i] < 0)outf << m_id_to_string[i] << " " << i << std::endl;
			else outf << m_id_to_string[i] << " " << i << " " << m_freqs[i] << std::endl;
		}
	}

	void initial(const unordered_map<string, int>& elem_stat, int cutOff = 0){
		clear();
		static unordered_map<string, int>::const_iterator elem_iter;
		static vector<pair<string, int> > t_vec;

		t_vec.clear();
		for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
			if (elem_iter->second > cutOff) {
				t_vec.push_back(make_pair(elem_iter->first, elem_iter->second));
			}
		}
		std::sort(t_vec.begin(), t_vec.end(), cmpStringIntPairByValue);

		for (int idx = 0; idx < t_vec.size(); idx++) {
			from_string(t_vec[idx].first, t_vec[idx].second);
		}
		t_vec.clear();
		set_fixed_flag(true);
	}

	//ugly implemented
	int highfreq(){		
		if (m_size <= 512) return m_size;

		blong sum = 0;
		for (int idx = 0; idx < m_size; idx++){
			sum += m_freqs[idx];
		}

		blong reserved = (blong)(sum * 0.8);
		int answer = 0;
		sum = 0;
		while (answer < m_size){
			sum += m_freqs[answer];
			if (sum >= reserved){
				break;
			}
			answer++;
		}

		return answer;
	}
};

typedef basic_quark Alphabet;
typedef basic_quark*  PAlphabet;

#endif

