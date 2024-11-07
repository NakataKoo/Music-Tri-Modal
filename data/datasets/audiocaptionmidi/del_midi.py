import os

data = [
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/Y/X/TRTYXZA128F42A4881.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/S/R/TRTSRTJ128F4297FFB.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/T/F/E/TRTFEXB128F92D58E2.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/T/J/TRWTJBI12903CD8457.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/H/Z/TRWHZSQ128F92EDE87.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/S/D/TRWSDDE128F423B10A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/C/Q/TRWCQTZ12903CC5BA1.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/U/W/TRWUWZV128F92E5796.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/V/S/TRWVSRD128F92D223F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/B/V/TRWBVNO128F92FB9DC.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/B/R/TRWBRDE128F425A470.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/I/H/TRWIHRP12903D0706D.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/F/R/TRWFRWI12903CEE074.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/Z/F/TRWZFNY128F422B6B2.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/W/Y/TRLWYMH128F424C7DA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/W/X/TRLWXSC128F4229174.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/M/H/TRLMHJL12903CC6D50.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/Y/Q/TRLYQIX128F4264BE6.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/S/H/TRLSHPW128F42AB061.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/J/S/TRLJSMW12903CD6B5C.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/V/T/TRLVTWI128F92EFEA7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/L/R/M/TRLRMBJ128F92E2FCD.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/T/P/TRATPFO128F42A3A47.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/A/M/TRAAMBM128F4248306.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/J/S/TRAJSAN128F426B1D3.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/V/D/TRAVDNS128F92F7B7F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/I/U/TRAIUAJ128F145EEDA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/F/J/TRAFJJC128F42A2E05.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/X/J/TRAXJZN128F42482FD.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/A/E/E/TRAEEBP128EF3673A7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/L/S/TRPLSQD128F428C573.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/Y/P/TRPYPTV128F9311C06.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/S/K/TRPSKFO12903CA1410.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/D/F/TRPDFOL12903D1525E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/C/T/TRPCTOB128F92F9676.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/C/A/TRPCAOD128E0788980.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/U/O/TRPUOVN128F92EE1AE.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/P/E/L/TRPELXZ12903CE2F40.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/Y/P/TRMYPVQ12903CDA4DB.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/Y/X/TRMYXYX128F422B792.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/J/E/TRMJEOS12903CE368D.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/U/A/TRMUAMW128F92E1415.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/A/TRMVANR128F42827D3.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/S/TRMVSKB128F426F411.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/V/F/TRMVFYZ128F428CBB4.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/B/U/TRMBUFU128F42738AD.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/I/F/TRMIFXJ128F42610E2.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/F/K/TRMFKYL12903CE2D2C.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/P/N/TRQPNQH128F42971DD.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/M/Z/TRQMZXY128F427AC9A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/Q/B/TRQQBLD128F148AA19.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/Q/Z/TRQQZJP128EF3559DF.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Q/O/D/TRQODUP128F93446A2.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/W/G/TRYWGDO128F92C5B81.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/A/Z/TRYAZYK128F931C1A5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/P/Y/TRYPYNT128F932F56E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/Y/X/TRYYXEY12903CB1EFB.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/K/Y/TRYKYUI128F42B47F3.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/B/X/TRYBXXD128F92F0DE8.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/G/P/TRYGPYN128F427B1D7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/Z/E/TRYZEEL128F92DE9AC.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/Q/Z/TRKQZNL128F92E11A6.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/S/W/TRKSWEI128F92ED8B0.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/S/F/TRKSFKR128F422006E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/N/V/TRKNVEI12903CDDEDA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/V/W/TRKVWBN12903CF5C45.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/E/Y/TRKEYDI128E079107F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/W/C/TRHWCPE128F4265BA5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/P/S/TRHPSXO128F92FA1C3.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/K/W/TRHKWAO128F92EA2EA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/D/G/TRHDGDU128F92EA54D.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/C/G/TRHCGLW12903CA4A0F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/V/T/TRHVTKV128F92F5EC1.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/Z/R/TRHZRTY128F932F5D3.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/H/E/G/TRHEGTN128F4278401.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/W/R/TRSWRPD128F422EE52.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/P/J/TRSPJJB128E078AF34.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/K/C/TRSKCJK128E07886B1.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/C/Z/TRSCZJC128F425F8A0.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/B/C/TRSBCXR128F92E1B1B.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/R/A/TRSRAMG12903CAF964.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/A/I/TRNAIFM128F1494318.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/A/E/TRNAEKI12903CAAD90.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/Q/L/TRNQLIG128F427DC4F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/S/Y/TRNSYON128F14A111A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/M/TRNIMOG128F4297583.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/Y/TRNIYQC128F931E626.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/I/O/TRNIOTS128F424E2F7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/G/O/TRNGOWG12903D1524E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/Z/Y/TRNZYLN128E078BE42.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/N/E/H/TRNEHHA12903CD2E92.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/L/F/TRDLFYP12903CE9219.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/K/S/TRDKSKE128F930A28A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/S/W/TRDSWOK128F933FCFE.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/O/U/TRDOUVT128F1464F30.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/B/F/TRDBFIY128E07937A6.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/O/TRDGOJH128F148CCDB.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/U/TRDGUAJ128F92D541D.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/D/G/X/TRDGXOS128F146A965.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/P/Y/TROPYIN128F422719E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/K/E/TROKERH128F92E587A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/H/E/TROHEKB128F428C56A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/N/E/TRONEAX128F9316517.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/O/U/F/TROUFUC128F145DB88.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/T/K/TRCTKPB12903CB8C53.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/W/S/TRCWSHF128F422B674.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/P/N/TRCPNVK128F4276004.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/Q/A/TRCQAND128F14B0478.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/N/Y/TRCNYXC128F42A0728.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/N/C/TRCNCSS128F932F5C5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/C/B/TRCCBEL12903CE88C8.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/B/Q/TRCBQXD128F14824E6.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/Z/M/TRCZMLT128F4248307.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/C/E/F/TRCEFKJ128F427A435.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/Q/O/TRJQOHN128F92D9DF7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/J/J/TRJJJCW128F42218C7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/F/L/TRJFLCS12903CB2B58.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/F/Y/TRJFYVA12903CFC456.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/J/X/N/TRJXNQV128F4297581.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/L/D/TRULDKO12903D1113E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/L/U/TRULUTD128F428564B.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/P/D/TRUPDNU128F147CBE9.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/P/I/TRUPIIH128F1489D45.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/E/S/TRUESTL128E0791A5B.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/G/G/TRVGGYU128F930A9FA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/R/C/TRVRCAN128F92C81DA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/E/U/TRVEUWZ128F42684C5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Y/S/TRBYSLP128F147CC94.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Y/J/TRBYJCT12903CF1F12.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/C/Q/TRBCQLR12903D11151.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/G/S/TRBGSBH128F930B3B7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/B/Z/P/TRBZPFH128F92F0DEA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/P/Z/TRIPZPZ128F4279B47.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/Y/K/TRIYKFK128F425E16F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/B/P/TRIBPQX128F4291617.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/G/H/TRIGHLA128F42B82E1.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/F/R/TRIFRCS128F92EA750.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Q/I/TRGQICW128F1492944.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Y/W/TRGYWIU128F1468031.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/S/Z/TRGSZLI128F4230F3A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/R/C/TRGRCFQ128F146DE19.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Z/Y/TRGZYUD128F9316BBA.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/G/Z/Z/TRGZZUV128F1463957.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/T/O/TRFTOHW12903D079F2.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/L/Y/TRFLYMX128F426EC7B.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/M/A/TRFMATK128F930291A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/F/Y/B/TRFYBKR128F4297574.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/H/X/TRRHXTQ128F428A76E.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/N/C/TRRNCJG12903CFD79A.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/O/N/TRRONGH128F931E1C0.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/U/A/TRRUAYW128EF357F83.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/I/U/TRRIUKV128F423AAF5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/X/R/TRRXRJP128F92F302C.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/A/H/TRZAHFW128F93414E4.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/D/D/TRZDDXA128F1455851.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/D/R/TRZDRJA128E0784714.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/G/G/TRZGGHO128F423F62F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/W/Q/TRXWQIP128F425FF5C.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/W/E/TRXWEJW128E0788449.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/P/C/TRXPCNN128F4277EF5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/D/M/TRXDMCF128F422005F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/J/B/TRXJBOT128F1496AD4.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/J/G/TRXJGEA12903CE77F7.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/U/Z/TRXUZRN128F4298801.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/B/H/TRXBHWC128F426E437.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/B/J/TRXBJWO128F4228D9F.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/X/C/TRXXCFL128F145E5AC.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/E/V/TRXEVRI128F4231565.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/A/N/TREANVZ12903CD73CB.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/P/C/TREPCOR128F424D5C5.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/M/L/TREMLJU128F422311B.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/S/Y/TRESYDL128E0787919.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/C/F/TRECFED128F932F506.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/Z/L/TREZLQY128F424EB04.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/Z/J/TREZJKA128F422BA79.npy"
    },
    {
        "audio_path": "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/E/X/P/TREXPXX128F4276361.npy"
    }
]

file_paths = [item["audio_path"] for item in data]

file_paths = file_paths+["/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/W/M/F/TRWMFBB128F4276455.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/I/P/TRMIPNS128F931EDF8.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/M/R/F/TRMRFZK128F92D14E2.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/A/L/TRYALCZ128F427DC4B.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Y/S/H/TRYSHSW128F427DC46.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/K/D/S/TRKDSMO128F93367EF.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/S/U/B/TRSUBFN128F931EDFE.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/U/N/Y/TRUNYCK12903CBB723.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/V/I/Y/TRVIYKM12903CBB725.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/I/S/V/TRISVQO128F4276451.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/R/U/S/TRRUSSW128F931AD36.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/W/J/TRZWJOO128F92DBD64.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/V/I/TRZVISD128F9306410.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/G/H/TRZGHVT12903CBB60E.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/Z/X/N/TRZXNZF128F931EDF3.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/Y/V/TRXYVYU12903CBB71D.npy", "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/X/Z/P/TRXZPKG12903CBB728.npy"]

# ファイルが存在するか確認し、存在すれば削除する
for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} を削除しました。")
    else:
        print(f"{file_path} が存在しません。")