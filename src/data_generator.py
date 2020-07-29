import data_converters
import tensorflow as tf


def conll_data_generator(filenames, data_config):
    """
    Read CoNLL formated @filenames files. Yields each sentence.
    Select columns defined in @data_config. data_config optionaly specify
    converters.
    """
    for filename in filenames:
        with open(filename, 'r') as f:
            sents = 0
            toks = 0
            buf = []
            for line in f:
                    line = line.strip()
                    if line:
                        toks += 1
                        split_line = line.split()
                        data_vals = []
                        for d in data_config.keys():
                                # only return the data that we're actually going to use as inputs or outputs
                                if (('feature' in data_config[d]
                                        and data_config[d]['feature'])
                                    or ('label' in data_config[d]
                                        and data_config[d]['label'])):

                                    datum_idx = data_config[d]['conll_idx']
                                    converter_name = data_config[d]['converter']['name'] if 'converter' in data_config[d] else 'default_converter'
                                    converter_params = data_converters.get_params(
                                        data_config[d],
                                        split_line, datum_idx)
                                    tf.logging.log(
                                        tf.logging.INFO,
                                        f"conll_data_generator dispatching for {d}: "
                                        f"{converter_name}, "
                                        f"{converter_params}")
                                    data = data_converters.dispatch(
                                        converter_name)(**converter_params)
                                    data_vals.extend(data)
                        # print(tuple(data_vals))
                        buf.append(tuple(data_vals))
                    else:
                        if buf:
                            sents += 1
                            tf.logging.log(
                                tf.logging.INFO,
                                f"data_generator.conll_data_generator "
                                f"yielding buf: {buf}: ")
                            yield buf
                            buf = []
            # catch the last one
            if buf:
                tf.logging.log(
                    tf.logging.INFO,
                    f"data_generator.conll_data_generator "
                    f"yielding last buf from {filename}: {buf}: ")
                yield buf
