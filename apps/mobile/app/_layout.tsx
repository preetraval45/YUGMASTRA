import { Stack } from 'expo-router';
import { Provider as PaperProvider } from 'react-native-paper';

export default function RootLayout() {
  return (
    <PaperProvider>
      <Stack>
        <Stack.Screen
          name="index"
          options={{
            title: 'YUGMĀSTRA',
            headerStyle: {
              backgroundColor: '#1e40af',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
        />
        <Stack.Screen
          name="dashboard"
          options={{
            title: 'Dashboard',
            headerStyle: {
              backgroundColor: '#1e40af',
            },
            headerTintColor: '#fff',
          }}
        />
      </Stack>
    </PaperProvider>
  );
}
