import os

data = [
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/Y/X/TRTYXZA128F42A4881.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/S/R/TRTSRTJ128F4297FFB.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/F/E/TRTFEXB128F92D58E2.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/T/J/TRWTJBI12903CD8457.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/H/Z/TRWHZSQ128F92EDE87.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/S/D/TRWSDDE128F423B10A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/C/Q/TRWCQTZ12903CC5BA1.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/U/W/TRWUWZV128F92E5796.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/V/S/TRWVSRD128F92D223F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/B/V/TRWBVNO128F92FB9DC.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/B/R/TRWBRDE128F425A470.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/I/H/TRWIHRP12903D0706D.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/F/R/TRWFRWI12903CEE074.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/Z/F/TRWZFNY128F422B6B2.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/W/Y/TRLWYMH128F424C7DA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/W/X/TRLWXSC128F4229174.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/M/H/TRLMHJL12903CC6D50.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/Y/Q/TRLYQIX128F4264BE6.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/S/H/TRLSHPW128F42AB061.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/J/S/TRLJSMW12903CD6B5C.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/V/T/TRLVTWI128F92EFEA7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/R/M/TRLRMBJ128F92E2FCD.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/T/P/TRATPFO128F42A3A47.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/A/M/TRAAMBM128F4248306.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/J/S/TRAJSAN128F426B1D3.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/V/D/TRAVDNS128F92F7B7F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/I/U/TRAIUAJ128F145EEDA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/F/J/TRAFJJC128F42A2E05.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/X/J/TRAXJZN128F42482FD.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/E/E/TRAEEBP128EF3673A7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/L/S/TRPLSQD128F428C573.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/Y/P/TRPYPTV128F9311C06.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/S/K/TRPSKFO12903CA1410.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/D/F/TRPDFOL12903D1525E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/C/T/TRPCTOB128F92F9676.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/C/A/TRPCAOD128E0788980.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/U/O/TRPUOVN128F92EE1AE.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/E/L/TRPELXZ12903CE2F40.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/Y/P/TRMYPVQ12903CDA4DB.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/Y/X/TRMYXYX128F422B792.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/J/E/TRMJEOS12903CE368D.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/U/A/TRMUAMW128F92E1415.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/A/TRMVANR128F42827D3.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/S/TRMVSKB128F426F411.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/F/TRMVFYZ128F428CBB4.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/B/U/TRMBUFU128F42738AD.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/I/F/TRMIFXJ128F42610E2.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/F/K/TRMFKYL12903CE2D2C.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/P/N/TRQPNQH128F42971DD.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/M/Z/TRQMZXY128F427AC9A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/Q/B/TRQQBLD128F148AA19.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/Q/Z/TRQQZJP128EF3559DF.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/O/D/TRQODUP128F93446A2.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/W/G/TRYWGDO128F92C5B81.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/A/Z/TRYAZYK128F931C1A5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/P/Y/TRYPYNT128F932F56E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/Y/X/TRYYXEY12903CB1EFB.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/K/Y/TRYKYUI128F42B47F3.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/B/X/TRYBXXD128F92F0DE8.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/G/P/TRYGPYN128F427B1D7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/Z/E/TRYZEEL128F92DE9AC.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/Q/Z/TRKQZNL128F92E11A6.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/S/W/TRKSWEI128F92ED8B0.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/S/F/TRKSFKR128F422006E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/N/V/TRKNVEI12903CDDEDA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/V/W/TRKVWBN12903CF5C45.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/E/Y/TRKEYDI128E079107F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/W/C/TRHWCPE128F4265BA5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/P/S/TRHPSXO128F92FA1C3.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/K/W/TRHKWAO128F92EA2EA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/D/G/TRHDGDU128F92EA54D.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/C/G/TRHCGLW12903CA4A0F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/V/T/TRHVTKV128F92F5EC1.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/Z/R/TRHZRTY128F932F5D3.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/E/G/TRHEGTN128F4278401.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/W/R/TRSWRPD128F422EE52.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/P/J/TRSPJJB128E078AF34.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/K/C/TRSKCJK128E07886B1.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/C/Z/TRSCZJC128F425F8A0.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/B/C/TRSBCXR128F92E1B1B.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/R/A/TRSRAMG12903CAF964.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/A/I/TRNAIFM128F1494318.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/A/E/TRNAEKI12903CAAD90.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/Q/L/TRNQLIG128F427DC4F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/S/Y/TRNSYON128F14A111A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/M/TRNIMOG128F4297583.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/Y/TRNIYQC128F931E626.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/O/TRNIOTS128F424E2F7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/G/O/TRNGOWG12903D1524E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/Z/Y/TRNZYLN128E078BE42.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/E/H/TRNEHHA12903CD2E92.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/L/F/TRDLFYP12903CE9219.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/K/S/TRDKSKE128F930A28A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/S/W/TRDSWOK128F933FCFE.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/O/U/TRDOUVT128F1464F30.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/B/F/TRDBFIY128E07937A6.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/O/TRDGOJH128F148CCDB.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/U/TRDGUAJ128F92D541D.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/X/TRDGXOS128F146A965.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/P/Y/TROPYIN128F422719E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/K/E/TROKERH128F92E587A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/H/E/TROHEKB128F428C56A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/N/E/TRONEAX128F9316517.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/U/F/TROUFUC128F145DB88.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/T/K/TRCTKPB12903CB8C53.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/W/S/TRCWSHF128F422B674.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/P/N/TRCPNVK128F4276004.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/Q/A/TRCQAND128F14B0478.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/N/Y/TRCNYXC128F42A0728.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/N/C/TRCNCSS128F932F5C5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/C/B/TRCCBEL12903CE88C8.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/B/Q/TRCBQXD128F14824E6.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/Z/M/TRCZMLT128F4248307.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/E/F/TRCEFKJ128F427A435.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/Q/O/TRJQOHN128F92D9DF7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/J/J/TRJJJCW128F42218C7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/F/L/TRJFLCS12903CB2B58.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/F/Y/TRJFYVA12903CFC456.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/X/N/TRJXNQV128F4297581.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/L/D/TRULDKO12903D1113E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/L/U/TRULUTD128F428564B.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/P/D/TRUPDNU128F147CBE9.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/P/I/TRUPIIH128F1489D45.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/E/S/TRUESTL128E0791A5B.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/G/G/TRVGGYU128F930A9FA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/R/C/TRVRCAN128F92C81DA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/E/U/TRVEUWZ128F42684C5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Y/S/TRBYSLP128F147CC94.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Y/J/TRBYJCT12903CF1F12.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/C/Q/TRBCQLR12903D11151.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/G/S/TRBGSBH128F930B3B7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Z/P/TRBZPFH128F92F0DEA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/P/Z/TRIPZPZ128F4279B47.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/Y/K/TRIYKFK128F425E16F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/B/P/TRIBPQX128F4291617.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/G/H/TRIGHLA128F42B82E1.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/F/R/TRIFRCS128F92EA750.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Q/I/TRGQICW128F1492944.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Y/W/TRGYWIU128F1468031.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/S/Z/TRGSZLI128F4230F3A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/R/C/TRGRCFQ128F146DE19.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Z/Y/TRGZYUD128F9316BBA.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Z/Z/TRGZZUV128F1463957.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/T/O/TRFTOHW12903D079F2.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/L/Y/TRFLYMX128F426EC7B.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/M/A/TRFMATK128F930291A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/Y/B/TRFYBKR128F4297574.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/H/X/TRRHXTQ128F428A76E.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/N/C/TRRNCJG12903CFD79A.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/O/N/TRRONGH128F931E1C0.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/U/A/TRRUAYW128EF357F83.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/I/U/TRRIUKV128F423AAF5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/X/R/TRRXRJP128F92F302C.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/A/H/TRZAHFW128F93414E4.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/D/D/TRZDDXA128F1455851.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/D/R/TRZDRJA128E0784714.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/G/G/TRZGGHO128F423F62F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/W/Q/TRXWQIP128F425FF5C.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/W/E/TRXWEJW128E0788449.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/P/C/TRXPCNN128F4277EF5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/D/M/TRXDMCF128F422005F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/J/B/TRXJBOT128F1496AD4.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/J/G/TRXJGEA12903CE77F7.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/U/Z/TRXUZRN128F4298801.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/B/H/TRXBHWC128F426E437.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/B/J/TRXBJWO128F4228D9F.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/X/C/TRXXCFL128F145E5AC.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/E/V/TRXEVRI128F4231565.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/A/N/TREANVZ12903CD73CB.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/P/C/TREPCOR128F424D5C5.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/M/L/TREMLJU128F422311B.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/S/Y/TRESYDL128E0787919.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/C/F/TRECFED128F932F506.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/Z/L/TREZLQY128F424EB04.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/Z/J/TREZJKA128F422BA79.mid"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/X/P/TREXPXX128F4276361.mid"
    }
]

file_paths = [item["audio_path"].replace(".mid", ".mp3").replace("/midi/audio2midi", "/audio/lmd_matched_mp3") for item in data]

file_paths = file_paths+["/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/W/M/F/TRWMFBB128F4276455.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/M/I/P/TRMIPNS128F931EDF8.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/M/R/F/TRMRFZK128F92D14E2.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Y/A/L/TRYALCZ128F427DC4B.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Y/S/H/TRYSHSW128F427DC46.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/K/D/S/TRKDSMO128F93367EF.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/S/U/B/TRSUBFN128F931EDFE.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/U/N/Y/TRUNYCK12903CBB723.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/V/I/Y/TRVIYKM12903CBB725.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/I/S/V/TRISVQO128F4276451.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/R/U/S/TRRUSSW128F931AD36.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Z/W/J/TRZWJOO128F92DBD64.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Z/V/I/TRZVISD128F9306410.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Z/G/H/TRZGHVT12903CBB60E.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/Z/X/N/TRZXNZF128F931EDF3.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/X/Y/V/TRXYVYU12903CBB71D.mp3", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/X/Z/P/TRXZPKG12903CBB728.mp3"]

# ファイルが存在するか確認し、存在すれば削除する
for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} を削除しました。")
    else:
        print(f"{file_path} が存在しません。")