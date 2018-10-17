#!/bin/sh
dev_clean_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_dev_clean*/result.txt  | awk '{print $11}' | tail -n 1)
test_clean_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_clean*/result.txt  | awk '{print $11}' | tail -n 1)

dev_clean_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_dev_clean*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
test_clean_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_clean*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
echo WER-clean dev $dev_clean_wer % test $test_clean_wer %

dev_other_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_dev_other*/result.txt  | awk '{print $11}' | tail -n 1)
test_other_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_other*/result.txt  | awk '{print $11}' | tail -n 1)

dev_other_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_dev_other*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
test_other_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_other*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
echo WER-other dev $dev_other_wer % test $test_other_wer %



echo "| dev_clean CER  | test_clean CER  | dev_clean WER | test_clean WER | dev_other CER  | test_other CER  | dev_other WER | test_other WER | path "
echo "| $dev_clean_cer | $test_clean_cer | $dev_clean_wer | $test_clean_wer | $dev_other_cer | $test_other_cer | $dev_other_wer | $test_other_wer | $1 "
