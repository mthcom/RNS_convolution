class sc_convertor():
	sn_cache = dict()
	def convert_to_binary(dec_number, bits):
		binary = bin(dec_number)[2:]
		while(len(binary) < bits):
			binary = '0' + binary
		result = list()
		for i in binary:
			result.append(int(i))
		return result
	def wbg(counter_out, binary_number):
		for i in range(len(counter_out)):
			before_wires = counter_out[:i]
			before_ands = 1
			for j in before_wires:
				before_ands = before_ands & ~j
			before_ands = before_ands & counter_out[i]
			before_ands = before_ands & binary_number[i]
			if(before_ands == 1):
				return 1
		return 0
	def get_wbg_stream_of_bits(bits, bit_index): #bit_index starts from 0
		# todo: add cache
		result = 0
		idx = bit_index + 1
		part_len = 2 ** (2 ** idx - 1)
		for i in range(2 ** (bits - bit_index - 1)):
			result = result | part_len << ((2 ** idx) * i)
		result = result >> (2 ** (idx - 1))
		return result
	def convert_to_sc(bits, number):
		if(number in sc_convertor.sn_cache):
			return sc_convertor.sn_cache[number]
		result = 0
		for i in range(bits):
			idx = bits - i - 1
			if(((2 ** (idx)) & number) >> (idx) == 1):
				result = result | sc_convertor.get_wbg_stream_of_bits(bits, i)
		sc_convertor.sn_cache[number] = result
		return result
	def skippy_mult(f, w, bits):
	    if(w > 1):
	        raise ValueError('Weight is not between 0 and 1')
	    sc_f = sc_convertor.convert_to_sc(bits, f)
	    sc_w = int((2 ** bits) * w)
	    result = 0
	    for i in range(sc_w):
	    	result += (sc_f) & 1
	    	sc_f = sc_f >> 1
	    return result
	def clear_cache():
	    sc_convertor.sn_cache.clear()
	def new_mult(f, w, bits):
		sc_f = sc_convertor.convert_to_sc(bits, f)
		sc_w = sc_convertor.convert_to_sc(bits, w)
		result = 0
		this_w = sc_w
		for i in range(2 ** bits):
			this_and = sc_f & this_w
			for j in range(2 ** bits):
				result += this_and & 1
				this_and = this_and >> 1
			this_w = (this_w >> 1) | ((this_w & 1) << (2 ** bits - 1))
		return result
